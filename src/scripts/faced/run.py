"""Train/validation-only visual-attribute alignment pilot; NOT emotion evaluation.
No class labels enter the losses. Original 480-token cache is read-only.
"""
import sys, json, time, hashlib, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT=Path('/data/model/datasets/FACED_syn50614194')
DATA=ROOT/'official200_classification_v1'
CACHE=ROOT/'original480_anchor_v1'
OUT=ROOT/'traceable_visual_alignment_v1'
MODEL='/home/ubuntu/work/eeg_project/oneemo/weights'
BANDS=[('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]
CHANNELS='Fp1 Fp2 Fz F3 F4 F7 F8 FC1 FC2 FC5 FC6 Cz C3 C4 T7 T8 CP1 CP2 CP5 CP6 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2'.split()

def save(name,d):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def atlas(values,weights,limits):
    # Fixed band-specific TRAIN quantiles, never per-image autoscale.
    maps=(values@weights.T).reshape(5,16,16)
    z=np.clip((maps-limits[0,:,None,None])/(limits[1]-limits[0])[:,None,None],0,1)
    rgb=np.stack([z,1-np.abs(2*z-1),1-z],-1)
    image=Image.new('RGB',(320,96),'white');draw=ImageDraw.Draw(image)
    for k,(name,_,_) in enumerate(BANDS):
        tile=Image.fromarray(np.uint8(np.rint(rgb[k]*255))).resize((64,64),Image.Resampling.NEAREST)
        image.paste(tile,(64*k,16));draw.text((64*k+2,1),name,fill='black')
    draw.text((2,81),'fixed scale; front at top; scalp 30ch',fill='black')
    return image

class AttributeAligner(nn.Module):
    def __init__(self,n,d):
        super().__init__()
        self.input=nn.Linear(64,128)
        self.query=nn.Parameter(torch.randn(1,n,128)*.02)
        self.att=nn.MultiheadAttention(128,4,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,d))
        self.measure=nn.Sequential(nn.Flatten(),nn.Linear(48*64,256),nn.GELU(),nn.Linear(256,150))
    def forward(self,x):
        k=self.input(x);q=self.query.expand(len(x),-1,-1)
        v,_=self.att(q,k,k,need_weights=False)
        return self.out(q+v),self.measure(x).reshape(-1,5,30)

def main():
    OUT.mkdir(exist_ok=False);start=time.time()
    torch.set_num_threads(4);torch.manual_seed(20261106)
    rng=np.random.default_rng(20261106)
    prepared=json.loads((DATA/'prepared.json').read_text())
    assert prepared['test_samples_read'] is False
    # Two random training segments per training subject; four per validation subject.
    train=np.concatenate([s*84+rng.choice(84,2,replace=False) for s in range(80)])
    val=np.concatenate([s*84+rng.choice(84,4,replace=False) for s in range(80,100)])
    ids=np.r_[train,val];assert max(ids)<8400
    logs=np.load(DATA/'band_logpower.npy',mmap_mode='r')
    weights=np.load(DATA/'topology_geometry.npz')['weights'].astype('float32')
    # Fixed display limits fit all training rows only, not the pilot validation data.
    fit=np.asarray(logs[:6720]);limits=np.quantile(fit,[.01,.99],axis=(0,1,3)).astype('float32')
    stats=np.load(DATA/'topology_standardizer.npz')
    measures=np.array(logs[ids],copy=True)
    targets=((measures-stats['median'][None,None,:,None])/stats['scale'][None,None,:,None]).reshape(-1,5,30)
    # Independently recompute 18 training rows from official loader tensors.
    signal=np.load(DATA/'signal.npy',mmap_mode='r')
    uv=np.asarray(signal[train[:18],:30],dtype='float64')*100
    uv-=uv.mean(-1,keepdims=True);win=np.hanning(200);freq=np.fft.rfftfreq(200,1/200)
    psd=abs(np.fft.rfft(uv*win,axis=-1))**2/(200*np.sum(win**2));psd[...,1:-1]*=2
    power=np.stack([psd[...,(freq>=lo)&(freq<hi)].sum(-1) for _,lo,hi in BANDS],-1)
    replay=np.log10(power+1e-6).transpose(0,2,3,1).astype('float32')
    err=float(np.max(abs(replay-measures[:18])));assert err<1e-4,err
    records=[]
    with (OUT/'evidence.jsonl').open('w') as f:
        for row,i in enumerate(ids):
            for t in range(10):
                record={'id':f'S{int(i)//84:03d}_T{(int(i)%84)//3+1:02d}_G{int(i)%3}_W{t}',
                    'index':int(i),'subject':int(i)//84,'trial':(int(i)%84)//3+1,'segment':int(i)%3,
                    'window_s':[int(i)%3*10+t,int(i)%3*10+t+1],
                    'split':'train' if row<len(train) else 'val','channels':CHANNELS,
                    'bands':BANDS,'log10_power_uV2_plus_epsilon':measures[row,t].tolist()}
                f.write(json.dumps(record)+'\n');records.append(record)
    for j in range(4):atlas(measures[j,0],weights,limits).save(OUT/f'train_example_{j}.png')
    np.savez_compressed(OUT/'measurements.npz',indices=ids,log_power=measures,limits=limits,
                        median=stats['median'],scale=stats['scale'])
    protocol={'scope':'stage1 native-vision feature and measurement alignment pilot; no emotion/LLM ability claim',
        'train_segments':train.tolist(),'val_segments':val.tolist(),'train_windows':len(train)*10,
        'val_windows':len(val)*10,'measurement_source':prepared,
        'measurement_replay_18_train_max_abs_error':err,
        'render':'16x16 fixed interpolation -> nearest64x64; 5 fixed panels; RGB triangular ramp; front-at-top inherited geometry; no per-sample autoscale',
        'display_limits':{'quantiles':[.01,.99],'fit':'all6720 training segments, axes segment,time,channel','clip':'display only; numerical targets unclipped'},
        'numeric_target':'(log10(power_uV2+1e-6)-train_band_median)/train_band_IQR; floor1e-6',
        'alignment':'native OneEmo get_image_features pooler_output after visual merger; subtract per-position train mean, divide train global residual RMS',
        'signal':'read-only frozen original480 tokens; auxiliary per-window48 tokens (32 signal+16 topology); main480 unchanged',
        'loss':'visual residual MSE + 0.2 standardized measurement MSE; no category labels',
        'training':{'steps':1200,'batch':32,'optimizer':'AdamW','lr':.0003,'weight_decay':.01,'clip':1,'validation_interval':200},
        'limitations':['visual feature matching is not visual reading correctness','local source is an input address, not exclusive receptive field; expert tokens mix context','power maps are derived views, not independent observations','pilot sampling does not reproduce full classification benchmark'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_read':False}
    save('protocol.json',protocol);save('status.json',{'phase':'native_visual_extraction','images':len(records)})
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    lm=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
    for p in lm.parameters():p.requires_grad_(False)
    cache=None
    with torch.no_grad():
        for st in range(0,len(records),8):
            images=[atlas(measures[j//10,j%10],weights,limits) for j in range(st,min(st+8,len(records))) ]
            inp=processor.image_processor(images=images,return_tensors='pt',min_pixels=320*96,max_pixels=320*96)
            output=lm.get_image_features(pixel_values=inp['pixel_values'].cuda(),image_grid_thw=inp['image_grid_thw'].cuda(),return_dict=True)
            features=torch.stack(list(output.pooler_output)).float().cpu().numpy()
            assert features.ndim==3 and np.isfinite(features).all()
            if cache is None:
                cache=np.lib.format.open_memmap(OUT/'native_visual.npy',mode='w+',dtype='float16',shape=(len(records),*features.shape[1:]))
                protocol['visual_shape']=list(cache.shape);protocol['processor_grid']=inp['image_grid_thw'].tolist();save('protocol.json',protocol)
            cache[st:st+len(features)]=features
            if st%160==0:save('status.json',{'phase':'native_visual_extraction','done':st+len(features),'total':len(records)})
    cache.flush();del cache,lm,processor;torch.cuda.empty_cache()
    # Main features remain unchanged on disk; gather by exact time addresses.
    raw=np.load(CACHE/'frozen_tokens.npy',mmap_mode='r')
    h=np.array(raw[ids]);h=np.concatenate([h[:,:320].reshape(-1,32,10,64).transpose(0,2,1,3),h[:,320:].reshape(-1,10,16,64)],2).reshape(-1,48,64)
    nt=len(train)*10;xscale=float(np.sqrt(np.mean(h[:nt]**2)))
    x=torch.tensor(h/xscale,device='cuda');truth=torch.tensor(targets,device='cuda')
    v=np.array(np.load(OUT/'native_visual.npy',mmap_mode='r'),dtype='float32')
    center=v[:nt].mean(0,keepdims=True);scale=float(np.sqrt(np.mean((v[:nt]-center)**2)));assert scale>1e-8
    y=torch.tensor((v-center)/scale,device='cuda');del v,h
    np.savez_compressed(OUT/'alignment_scaling.npz',center=center,scale=scale,input_rms=xscale)
    net=AttributeAligner(y.shape[1],y.shape[2]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
    history=[];best=float('inf');gen=torch.Generator(device='cuda').manual_seed(20261106)
    @torch.no_grad()
    def evaluate(step):
        net.eval();sums=np.zeros(4);count=0
        for st in range(nt,len(x),32):
            ix=torch.arange(st,min(st+32,len(x)),device='cuda');pred,m=net(x[ix])
            # Next selected segment, same subject and 1-second offset. Label-independent.
            local=ix-nt
            donor=nt+(local//40)*40+(local%40+10)%40
            values=[(pred-y[ix]).square().mean(),(y[ix]).square().mean(),
                    (pred-y[donor]).square().mean(),(m-truth[ix]).abs().mean()]
            sums+=np.array([float(a) for a in values])*len(ix);count+=len(ix)
        row=dict(zip(['visual_MSE','train_mean_visual_baseline_MSE','mismatch_visual_MSE','measurement_MAE_IQR_units'],(sums/count).tolist()))
        row.update(step=step,seconds=time.time()-start)
        net.train();return row
    history.append(evaluate(0));save('history.json',history)
    for step in range(1,1201):
        ix=torch.randint(nt,(32,),generator=gen,device='cuda');net.train();opt.zero_grad(set_to_none=True)
        pred,m=net(x[ix]);loss=(pred-y[ix]).square().mean()+.2*(m-truth[ix]).square().mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%200==0:
            row=evaluate(step);row['train_batch_loss']=float(loss.detach());history.append(row)
            save('history.json',history);save('status.json',{'phase':'alignment_training',**row});print(json.dumps(row),flush=True)
            if row['visual_MSE']<best:
                best=row['visual_MSE'];torch.save({'model':net.state_dict(),'protocol':protocol,'step':step},OUT/'best.pt')
    save('complete.json',{'phase':'complete','history':history,'best_validation_visual_MSE':best,'seconds':time.time()-start,'test_read':False,
        'next_gate':'verify native image reading/perturbation and local content alignment before language task training; this pilot alone proves neither emotion performance nor attribution'})
    save('status.json',{'phase':'complete'})

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():save('status.json',{'phase':'failed','error':traceback.format_exc()})
        raise
