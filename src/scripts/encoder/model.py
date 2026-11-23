import torch
from torch import nn

class PatchTopologyClassifier(nn.Module):
    def __init__(self,arm,positions,weights,d=64,fs=200,readout="mean"):
        super().__init__();self.arm=arm;self.d=d;self.fs=fs;self.readout=readout
        assert arm in ['signal','topology','dual','experts']
        self.use_signal=arm!='topology';self.use_topo=arm!='signal'
        self.register_buffer('positions',positions/positions[:30].norm(dim=1).max())
        self.register_buffer('weights',weights)
        self.time=nn.Embedding(10,d)
        if self.use_signal:
            self.signal_embed=nn.Sequential(nn.Conv1d(1,8,15,stride=4,padding=7),nn.GELU(),
                nn.AvgPool1d(4),nn.Flatten(),nn.Linear(((fs+3)//4//4)*8,d))
            self.channel_embed=nn.Linear(3,d)
        if self.use_topo:
            self.topo_embed=nn.Conv2d(5,d,kernel_size=4,stride=4)
            gy,gx=torch.meshgrid(torch.linspace(.8,-.8,4),torch.linspace(-.8,.8,4),indexing='ij')
            xy=torch.stack([gx,gy],-1)
            self.register_buffer('image_positions',xy.reshape(16,2))
            self.image_embed=nn.Linear(2,d)
        self.local_norm=nn.LayerNorm(d)
        if arm=='experts':
            self.temporal=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
            self.spatial=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
            self.cross=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
            self.gate=nn.Linear(d,3)
            self.out_norm=nn.LayerNorm(d)
        self.classifier=nn.Linear(d*(self.use_signal+self.use_topo),9)
        if readout=='full':self.classifier=QueryFlattenHead(d,480)
        assert readout in ('mean','full')
    def features(self,signal,band,return_trace=False):
        b=signal.shape[0];t=self.time.weight
        e=v=None
        if self.use_signal:
            e=self.signal_embed(signal.reshape(-1,1,self.fs)).reshape(b,32,10,self.d)
            e=self.local_norm(torch.nn.functional.gelu(e+self.channel_embed(self.positions)[None,:,None,:]+t[None,None]))
        if self.use_topo:
            maps=torch.einsum('btfc,pc->btfp',band,self.weights).reshape(b*10,5,16,16)
            v=self.topo_embed(maps).flatten(2).transpose(1,2).reshape(b,10,16,self.d)
            v=self.local_norm(torch.nn.functional.gelu(v+t[None,:,None,:]+self.image_embed(self.image_positions)[None,None]))
        trace={}
        if self.arm=='experts':
            def att(m,q,k=None):
                k=q if k is None else k
                return m(q,k,k,need_weights=False)[0]
            # Temporal: same sensor/image patch across ten windows.
            et=att(self.temporal,e.reshape(b*32,10,self.d)).reshape_as(e)
            vt=att(self.temporal,v.transpose(1,2).reshape(b*16,10,self.d)).reshape(b,16,10,self.d).transpose(1,2)
            # Spatial: same time, interactions within each representation.
            es_in=e.transpose(1,2).reshape(b*10,32,self.d)
            vs_in=v.reshape(b*10,16,self.d)
            es=att(self.spatial,es_in).reshape(b,10,32,self.d).transpose(1,2)
            vs=att(self.spatial,vs_in).reshape_as(v)
            # Cross-view: time-aligned queries only; no interaction across unmatched windows.
            ec=att(self.cross,es_in,vs_in).reshape(b,10,32,self.d).transpose(1,2)
            vc=att(self.cross,vs_in,es_in).reshape_as(v)
            def route(z,choices):
                scores=self.gate(z);keep=scores.topk(2,dim=-1).indices
                mask=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,keep,True)
                g=scores.masked_fill(~mask,float('-inf')).softmax(-1)
                delta=(torch.stack(choices,-2)*g.unsqueeze(-1)).sum(-2)
                return self.out_norm(z+delta),g
            e,ge=route(e,[et,es,ec]);v,gv=route(v,[vt,vs,vc])
            if return_trace:
                trace={'signal_gate':ge,'topology_gate':gv,'expert_names':['temporal','spatial','cross_view']}
        return e,v,trace
    def forward(self,signal,band,return_trace=False):
        e,v,trace=self.features(signal,band,return_trace)
        if self.readout=='mean':
            logits=self.classifier(torch.cat([e.mean((1,2)),v.mean((1,2))],-1))
        else:
            tokens=torch.cat([e.reshape(len(e),320,self.d),v.reshape(len(v),160,self.d)],1)
            logits=self.classifier(tokens)
        return (logits,trace) if return_trace else logits

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6).to(x.dtype)*self.weight
class QueryFlattenHead(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.query=nn.Parameter(torch.randn(1,1,d))
        self.norm=RMSNorm((n+1)*d);self.drop=nn.Dropout(.1);self.linear=nn.Linear((n+1)*d,9)
    def forward(self,tokens):
        q=self.query.expand(len(tokens),-1,-1)
        w=(q@tokens.transpose(-1,-2)/(tokens.shape[-1]**.5)).softmax(-1)
        context=w@tokens
        return self.linear(self.drop(self.norm(torch.cat([context,tokens],1).flatten(1))))
