"""Native topomap reader SFT: position and adjacent-window display change.
No emotional labels, EEG soft tokens or test subjects; new isolated output.
"""
from run import atlas,ROOT,DATA,MODEL,BANDS
import json,time,math,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForImageTextToText
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict

SOURCE=ROOT/'traceable_visual_alignment_v1'
OUT=ROOT/'traceable_visual_reader_sft_v1'
SEED=20261107
REGEX=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))'
LABELS={'position':['FRONT','BACK','SIMILAR'],'change':['INCREASE','DECREASE','SIMILAR']}

def save(name,d):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

def main():
    OUT.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED);z=np.load(SOURCE/'measurements.npz')
    ids=z['indices'];values=z['log_power'];limits=z['limits'];assert len(ids)==240 and max(ids)<8400
    weights=np.load(DATA/'topology_geometry.npz')['weights'].astype('float32')
    maps=(values@weights.T).reshape(240,10,5,16,16)
    displayed=np.clip((maps-limits[0][None,None,:,None,None])/(limits[1]-limits[0])[None,None,:,None,None],0,1)
    datasets={}
    for split,rows,cap in [('train',range(160),400),('val',range(160,240),40)]:
        groups={(task,label):[] for task,labs in LABELS.items() for label in labs}
        for r in rows:
            assert (int(ids[r])//84<80)==(split=='train')
            for t in range(10):
                for b in range(5):
                    a=displayed[r,t,b];delta=float(a[:8].mean()-a[8:].mean())
                    label='SIMILAR' if abs(delta)<=.03 else ('FRONT' if delta>0 else 'BACK')
                    groups['position',label].append(dict(task='position',target=label,row=r,window=t,band=b,delta=delta))
                    if t:
                        delta=float(a.mean()-displayed[r,t-1,b].mean())
                        label='SIMILAR' if abs(delta)<=.05 else ('INCREASE' if delta>0 else 'DECREASE')
                        groups['change',label].append(dict(task='change',target=label,row=r,window=t,band=b,delta=delta))
        n=min(cap,min(map(len,groups.values())));assert n>=10,(split,{str(k):len(v) for k,v in groups.items()})
        chosen=[]
        for group in groups.values():chosen.extend(group[j] for j in rng.choice(len(group),n,replace=False))
        rng.shuffle(chosen);datasets[split]=chosen
        with (OUT/f'{split}_records.jsonl').open('w') as f:
            for record in chosen:
                record.update(index=int(ids[record['row']]),subject=int(ids[record['row']])//84)
                f.write(json.dumps(record)+'\n')
    protocol={'scope':'supervised native visual reading only, not emotion or physiology inference',
        'train_n':len(datasets['train']),'val_n':len(datasets['val']),
        'source':str(SOURCE/'measurements.npz'),'source_sha256':hashlib.sha256((SOURCE/'measurements.npz').read_bytes()).hexdigest(),
        'split':'same pilot:160 train segments from subjects0-79;80 val segments subjects80-99; no test',
        'sampling':'balanced each task x3 displayed-attribute labels, max400 train/40 val per group; fixed seed; not natural prevalence',
        'labels':'deterministic clipped DISPLAY attributes, not emotion labels: position top/bottom mean gap0.03; change current-minus-previous mean gap0.05; inclusive SIMILAR thresholds',
        'reference':'immediately previous1second within SAME10second segment; no prior trial/person baseline',
        'images':'unchanged fixed-scale five-band atlas; no answer/source IDs embedded; native320x96 ->30 image tokens',
        'trainable':'native vision merger plus language attention LoRA rank16 alpha32 dropout0; all other vision/language/EEG frozen',
        'epochs':3,'microbatch':4,'accumulation':4,'effective_batch':16,'lr_lora':5e-5,'lr_merger':1e-5,
        'optimizer':'AdamW wd0 epsilon1e-6; linearwarmup5percent cosine; jointclip1',
        'loss':'autoregressive CE only assistant answer and chat suffix; prompt/padding labels=-100',
        'selection':'highest validation mean of task-wise3class BAcc; tie lower val CE; initial baseline excluded',
        'controls':'chosencheckpoint actual, fixed training-median maps, same-subject other selected segment; unchanged questions, targets',
        'limits':['small pilot; no emotion classification','display clipping can suppress raw numerical differences','CE remains for reading; no emotional category CE','source index is traceability not causal explanation'],
        'test_read':False,'seed':SEED,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save('protocol.json',protocol);save('status.json',{'phase':'loading_model'})
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);tok=processor.tokenizer
    lm=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    for p in lm.parameters():p.requires_grad_(False)
    lm=get_peft_model(lm,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=REGEX,bias='none',task_type='CAUSAL_LM'))
    merger=[];lora=[];merger_names=[]
    for n,p in lm.named_parameters():
        if '.visual.merger.' in n:p.requires_grad_(True);merger.append(p);merger_names.append(n)
        elif 'lora_' in n:lora.append(p)
    assert merger and lora
    lm.eval();lm.config.use_cache=False
    params=merger+lora
    protocol.update(trainable_parameters=sum(p.numel() for p in params),merger_parameter_names=merger_names)
    save('protocol.json',protocol)
    imagecache={}
    def picture(row,t,condition):
        if condition=='fixed':return atlas(np.broadcast_to(z['median'][:,None],(5,30)),weights,limits)
        if condition=='mismatch':
            size,base=(2,0) if row<160 else (4,160)
            row=base+(row-base)//size*size+((row-base)%size+1)%size
        key=(row,t)
        if key not in imagecache:imagecache[key]=atlas(values[row,t],weights,limits)
        return imagecache[key]
    def messages(record,condition):
        r,t,b=record['row'],record['window'],record['band'];band=BANDS[b][0]
        legend='Each image has five panels in order delta, theta, alpha, beta, gamma. Red is high power, blue low, green intermediate. Scales are fixed. '
        if record['task']=='position':
            prompt=legend+f'In the {band} panel, compare the mean displayed power of the upper(front) half with the lower(back) half. If the normalized difference is at most0.03 in magnitude, reply SIMILAR. Otherwise reply FRONT or BACK for the higher half. Reply exactly one label.'
            content=[{'type':'image'},{'type':'text','text':prompt}];images=[picture(r,t,condition)]
        else:
            prompt=legend+f'The first image is the immediately previous1second reference and the second is the current1second, from the same segment. In the {band} panel compare the mean displayed power across the full map. If current-minus-reference is within0.05 in normalized display units, reply SIMILAR. Otherwise reply INCREASE or DECREASE. Reply exactly one label.'
            content=[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}];images=[picture(r,t-1,condition),picture(r,t,condition)]
        return [{'role':'user','content':content}],images
    def inputs(records,answer=False,condition='actual'):
        texts=[];images=[];prompts=[]
        for r in records:
            ms,ims=messages(r,condition)
            prompt=processor.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompts.append(prompt)
            full=processor.apply_chat_template(ms+[{'role':'assistant','content':r['target']}],tokenize=False,add_generation_prompt=False,enable_thinking=False) if answer else prompt
            assert not answer or full.startswith(prompt)
            texts.append(full);images.extend(ims)
        tok.padding_side='right' if answer else 'left'
        kw=dict(images=images,return_tensors='pt',padding=True,min_pixels=320*96,max_pixels=320*96)
        batch=processor(text=texts,**kw)
        if answer:
            labels=batch['input_ids'].clone();prefix=processor(text=prompts,**kw)
            lengths=prefix['attention_mask'].sum(1).tolist()
            for j,n in enumerate(lengths):
                assert torch.equal(batch['input_ids'][j,:n],prefix['input_ids'][j,:n]),'Answer boundary mismatch'
                labels[j,:n]=-100
            labels[batch['attention_mask']==0]=-100;assert (labels!=-100).any(1).all();batch['labels']=labels
        return batch.to('cuda')
    def reset():
        base=lm.get_base_model()
        if hasattr(base.model,'rope_deltas'):base.model.rope_deltas=None
    def loss(records):
        batch=inputs(records,True);reset();return lm(**batch,use_cache=False).loss
    @torch.no_grad()
    def evaluate(tag,condition='actual'):
        lm.eval();records=datasets['val'];rows=[];totalce=0.
        for st in range(0,len(records),4):
            rr=records[st:st+4]
            if condition=='actual':totalce+=float(loss(rr))*len(rr)
            batch=inputs(rr,False,condition);reset();n=batch['input_ids'].shape[1]
            output=lm.generate(**batch,do_sample=False,max_new_tokens=12)
            texts=tok.batch_decode(output[:,n:],skip_special_tokens=True)
            for r,text in zip(rr,texts):
                pred=text.strip().upper().strip(' .\n\t')
                rows.append({**r,'answer':text,'prediction':pred if pred in LABELS[r['task']] else None})
        result={}
        for task,labs in LABELS.items():
            group=[r for r in rows if r['task']==task]
            recalls={lab:sum(r['target']==lab and r['prediction']==lab for r in group)/sum(r['target']==lab for r in group) for lab in labs}
            result[task]={'n':len(group),'bacc':sum(recalls.values())/3,'recall':recalls,
                'prediction_counts':{lab:sum(r['prediction']==lab for r in group) for lab in labs},'invalid':sum(r['prediction'] is None for r in group)}
        result['macro_bacc']=sum(result[t]['bacc'] for t in LABELS)/2
        if condition=='actual':result['ce']=totalce/len(records)
        with (OUT/f'{tag}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return result
    opt=torch.optim.AdamW([{'params':merger,'lr':1e-5},{'params':lora,'lr':5e-5}],weight_decay=0.,eps=1e-6)
    total=math.ceil(len(datasets['train'])/16)*3;warm=max(1,math.ceil(total*.05))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*min(1,(s-warm)/max(1,total-warm)))))
    # Real preflight; this does not update the model.
    opt.zero_grad();ls=loss(datasets['train'][:4]);ls.backward()
    mg=sum(float(p.grad.float().norm()) for p in merger if p.grad is not None)
    lg=sum(float(p.grad.float().norm()) for p in lora if p.grad is not None)
    assert mg>0 and lg>0 and all(p.grad is None for p in lm.parameters() if not p.requires_grad)
    save('preflight.json',{'ce':float(ls.detach()),'merger_gradient_norm_sum':mg,'lora_gradient_norm_sum':lg,'frozen_gradients_absent':True,'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9})
    opt.zero_grad();del ls
    save('status.json',{'phase':'initial_validation'});initial=evaluate('initial_validation');save('initial_validation.json',initial)
    history=[];best=(-1,float('-inf'));updates=0
    def checkpoint(ep):
        return {'lora':get_peft_model_state_dict(lm),'merger':{n:p.detach().cpu() for n,p in lm.named_parameters() if '.visual.merger.' in n},'epoch':ep,'updates':updates,'protocol':protocol}
    for ep in range(1,4):
        order=np.random.default_rng(SEED+ep).permutation(len(datasets['train']));online=0.;seen=0
        lm.eval()
        for st in range(0,len(order),16):
            ix=order[st:st+16];opt.zero_grad()
            for k in range(0,len(ix),4):
                rr=[datasets['train'][int(j)] for j in ix[k:k+4]];ls=loss(rr);assert torch.isfinite(ls)
                (ls*len(rr)/len(ix)).backward();online+=float(ls.detach())*len(rr);seen+=len(rr)
            gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();sched.step();updates+=1
            if updates%10==0:save('status.json',{'phase':'training','epoch':ep,'updates':updates,'total_updates':total,'seen':seen,'online_ce':online/seen,'gradient_norm':gn,'seconds':time.time()-start,'lr_lora':opt.param_groups[1]['lr']})
        ck=checkpoint(ep);torch.save({**ck,'optimizer':opt.state_dict(),'scheduler':sched.state_dict()},OUT/'last.pt')
        save('status.json',{'phase':'validation','epoch':ep,'updates':updates});vm=evaluate(f'validation_epoch_{ep:02d}')
        row={'epoch':ep,'updates':updates,'train_ce':online/seen,'validation':vm,'seconds':time.time()-start};history.append(row);save('history.json',history);print(json.dumps(row),flush=True)
        if (vm['macro_bacc'],-vm['ce'])>best:
            best=(vm['macro_bacc'],-vm['ce']);torch.save({**ck,'validation':vm},OUT/'best.pt')
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);set_peft_model_state_dict(lm,ck['lora'])
    with torch.no_grad():
        for n,p in lm.named_parameters():
            if n in ck['merger']:p.copy_(ck['merger'][n].to(p.device))
    actual=evaluate('best_actual');fixed=evaluate('best_fixed','fixed');mismatch=evaluate('best_mismatch','mismatch')
    save('complete.json',{'phase':'complete','initial':initial,'best_epoch':ck['epoch'],'actual':actual,'fixed':fixed,'mismatch':mismatch,'seconds':time.time()-start,'updates':updates,'test_read':False})
    save('status.json',{'phase':'complete'})

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():save('status.json',{'phase':'failed','traceback':traceback.format_exc()})
        raise
