"""Native topomap reader SFT: position and adjacent-window display change.
No emotional labels, EEG soft tokens or test subjects; new isolated output.
"""
from run import atlas,ROOT,DATA,MODEL,BANDS
import sys
import os
import json,time,math,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'encoder'))
from model import QueryFlattenHead
import numpy as np
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForImageTextToText
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict

SOURCE=ROOT/'traceable_visual_alignment_v1'
OUT=ROOT/'classifier_visual_multitask_v1'
REEVALUATE=os.environ.get('FACED_MULTITASK_REEVALUATE')=='1'
ORIGINAL_OUTPUT=OUT
if REEVALUATE:
    OUT=ROOT/'classifier_visual_multitask_v1_reeval'
SEED=20261108
REGEX=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))'
LABELS={'position':['FRONT','BACK','SIMILAR'],'change':['INCREASE','DECREASE','SIMILAR']}

def save(name,d):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

def main():
    OUT.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED);z=np.load(SOURCE/'measurements.npz')
    ids=z['indices'];values=z['log_power'];limits=z['limits'];assert len(ids)==240 and max(ids)<8400
    weights=np.load(DATA/'topology_geometry.npz')['weights'].astype('float32')
    cache_root=ROOT/'original480_anchor_v1'
    feature_cache=np.load(cache_root/'frozen_tokens.npy',mmap_mode='r')
    classifier_path=ROOT/'readout_recipe_2x2_v1/full_borrowed/best.pt'
    classifier_checkpoint=torch.load(classifier_path,map_location='cpu',weights_only=False)
    classifier=QueryFlattenHead(64,480).cuda().eval()
    classifier.load_state_dict({key.removeprefix('classifier.'):value for key,value in classifier_checkpoint['model'].items() if key.startswith('classifier.')})
    names=['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
    predictions={}
    with torch.no_grad():
        for offset in range(0,len(ids),64):
            indices=ids[offset:offset+64]
            features=torch.tensor(np.array(feature_cache[indices]),device='cuda')
            probability=classifier(features).softmax(-1).cpu().numpy()
            for index,prob in zip(indices,probability):
                predictions[int(index)]={'label':names[int(prob.argmax())],'probabilities':prob.tolist()}
    save('classifier_predictions.json',predictions)
    del classifier,classifier_checkpoint,features
    torch.cuda.empty_cache()
    maps=(values@weights.T).reshape(240,10,5,16,16)
    displayed=np.clip((maps-limits[0][None,None,:,None,None])/(limits[1]-limits[0])[None,None,:,None,None],0,1)
    datasets={};pair_counts={}
    for split,rows,cap in [('train',range(160),32),('val',range(160,240),8)]:
        candidates={(task,b):[] for task in LABELS for b in range(5)}
        for r in rows:
            assert (int(ids[r])//84<80)==(split=='train')
            for t in range(10):
                for b in range(5):
                    a=displayed[r,t,b];d=float(a[:8].mean()-a[8:].mean())
                    if abs(d)>=.06:candidates['position',b].append((r,t,d))
                    if t:
                        d=float(a.mean()-displayed[r,t-1,b].mean())
                        if abs(d)>=.08:candidates['change',b].append((r,t,d))
        n=min(cap,min(map(len,candidates.values())))
        assert n>=2,(split,{str(k):len(v) for k,v in candidates.items()})
        chosen=[];pair_counts[split]=n
        for (task,b),pool in candidates.items():
            for pick in rng.choice(len(pool),n,replace=False):
                r,t,d=pool[int(pick)];group=f'{split}_{task}_{b}_{r}_{t}'
                if task=='position':
                    specs=[('identity','FRONT' if d>0 else 'BACK'),('flip','BACK' if d>0 else 'FRONT'),('symmetric','SIMILAR')]
                else:
                    specs=[('identity','INCREASE' if d>0 else 'DECREASE'),('swap','DECREASE' if d>0 else 'INCREASE'),('same','SIMILAR')]
                for op,label in specs:
                    chosen.append(dict(task=task,band=b,row=r,window=t,delta=d,op=op,target=label,group=group,index=int(ids[r]),subject=int(ids[r])//84))
        rng.shuffle(chosen);datasets[split]=chosen
        with (OUT/f'{split}_records.jsonl').open('w') as f:
            for record in chosen:f.write(json.dumps(record)+'\n')
    # Pixel construction audits on training examples. Scalars refer to encoded colors.
    audits=[]
    for record in datasets['train'][:30]:
        r,t,b=record['row'],record['window'],record['band'];q=displayed[r,t,b]
        if record['task']=='position':
            flip=np.flip(q,0);sym=(q+flip)/2
            err=float(abs((q[:8].mean()-q[8:].mean())+(flip[:8].mean()-flip[8:].mean())))
            assert err<1e-5 and abs(float(sym[:8].mean()-sym[8:].mean()))<1e-5
            audits.append({'group':record['group'],'flip_sign_error':err})
    save('paired_render_preflight.json',{'training_only':True,'checks':audits,'per_band_task_base_groups':pair_counts,'test_read':False})
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
    initial_checkpoint=ROOT/'traceable_visual_reader_paired_v2/best.pt'
    protocol.update({'stage':'continue v1 reader with new balanced paired-image diagnostic; not equal-budget comparison',
        'initial_checkpoint':str(initial_checkpoint),'initial_checkpoint_sha256':hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest(),
        'sampling':'within EVERY task x band, balanced3labels from matched triples; max32 train/8 val base groups; same subject partition',
        'paired_position':'identity, vertical flip of queried band, vertical symmetrization of queried band; derived image intervention only',
        'paired_change':'natural adjacent windows, swap their input order, repeat reference image; same source group',
        'candidate_margin':'position abs(displaygap)>=0.06; change abs(displaygap)>=0.08, predeclared; easier separated diagnostic than v1',
        'labels':'same thresholds0.03/0.05; no labels rendered or supplied in prompt; every band task group uniform',
        'train_base_groups_per_band_task':pair_counts['train'],'val_base_groups_per_band_task':pair_counts['val'],
        'frozen':'all vision/language base except native merger and same language LoRA; no EEG input',
        'pair_metrics':'whole-triple all3correct; directional identity/transformed both correct; synthetic symmetric/same correct',
        'prior_baseline':'each task-band has exactly equal3labels; deterministic task-band-only predictor BAcc1/3'})
    protocol.update({
        'stage':'continue paired-v2 reader; structured classification handoff plus two visual-reading tasks',
        'classification':'frozen original480 QueryFlattenHead; outputs are model predictions, never self-report truth; no classifier updates',
        'classifier_checkpoint':str(classifier_path),
        'classifier_sha256':hashlib.sha256(classifier_path.read_bytes()).hexdigest(),
        'answer_format':'JSON emotion=<supplied classifier prediction>, evidence=<visual attribute label>',
        'training_tasks':'mixed spatial comparison and whole-map adjacent-window change; emotion copying included in each response',
        'final_class':'program locks category from classifier regardless of generated emotion; copy is separately audited',
        'causal_boundary':'synthetic paired visual comparisons are controlled rendering tasks, not physiological interventions or explanations of emotion',
        'reference':'previous one-second window within same segment; swap/same are synthetic comparisons',
        'selection':'mean spatial/change BAcc, tie lower CE; copying score not included in selection',
        'prior_baseline':'each evidence task balanced three labels; JSON format differs from reader-v2; initial validation recorded',
        'limits':['small controlled visual-reading pilot','change compares whole map, not localized regional change','no claim that observed power caused classifier decision','no final test evaluation']
    })
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
    initial=torch.load(initial_checkpoint,map_location='cpu',weights_only=False)
    set_peft_model_state_dict(lm,initial['lora'])
    with torch.no_grad():
        for n,p in lm.named_parameters():
            if n in initial['merger']:p.copy_(initial['merger'][n].to(p.device))
    del initial
    lm.eval();lm.config.use_cache=False
    params=merger+lora
    protocol.update(trainable_parameters=sum(p.numel() for p in params),merger_parameter_names=merger_names)
    save('protocol.json',protocol)
    imagecache={}
    def picture(row,t,condition,band=None,op='identity'):
        if condition=='mismatch':
            size,base=(2,0) if row<160 else (4,160)
            row=base+(row-base)//size*size+((row-base)%size+1)%size
        key=(row,t,condition,band,op)
        if key not in imagecache:
            if condition=='fixed':
                raw=(np.broadcast_to(z['median'][:,None],(5,30))@weights.T).reshape(5,16,16)
                q=np.clip((raw-limits[0,:,None,None])/(limits[1]-limits[0])[:,None,None],0,1)
            else:q=displayed[row,t].copy()
            if op=='flip':q[band]=np.flip(q[band],0)
            if op=='symmetric':q[band]=(q[band]+np.flip(q[band],0))/2
            # Quantize color after intervention; no target text or source IDs.
            rgb=np.stack([q,1-np.abs(2*q-1),1-q],-1)
            image=Image.new('RGB',(320,96),'white');draw=ImageDraw.Draw(image)
            for k,(name,_,_) in enumerate(BANDS):
                tile=Image.fromarray(np.uint8(np.rint(rgb[k]*255))).resize((64,64),Image.Resampling.NEAREST)
                image.paste(tile,(k*64,16));draw.text((k*64+2,1),name,fill='black')
            draw.text((2,81),'fixed scale; front at top; scalp 30ch',fill='black')
            imagecache[key]=image
        return imagecache[key]
    def messages(record,condition):
        r,t,b,op=record['row'],record['window'],record['band'],record['op'];band=BANDS[b][0]
        legend='Each image has five panels in order delta, theta, alpha, beta, gamma. Red is high power, blue low, green intermediate. Scales are fixed. '
        if record['task']=='position':
            prompt=legend+f'In the {band} panel, compare the mean displayed power of the upper(front) half with the lower(back) half. If the normalized difference is at most0.03 in magnitude, reply SIMILAR. Otherwise reply FRONT or BACK for the higher half. Reply exactly one label.'
            content=[{'type':'image'},{'type':'text','text':prompt}];images=[picture(r,t,condition,b,op)]
        else:
            # This is an image comparison; swaps are not called natural chronology.
            prompt=legend+f'The first image is the reference and the second image is the query. In the {band} panel compare the mean displayed power across the full map. If query-minus-reference is within0.05 in normalized display units, reply SIMILAR. Otherwise reply INCREASE or DECREASE. Reply exactly one label.'
            pair=[t-1,t] if op=='identity' else ([t,t-1] if op=='swap' else [t-1,t-1])
            content=[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}];images=[picture(r,tt,condition) for tt in pair]
        decision=predictions[record['index']]['label']
        content[-1]['text'] += (' A separate frozen classifier supplied classifier_label: '+decision+
            '. This is a model prediction, not ground truth or an explanation of the image. '
            'The diagram may be a controlled transformation. Do not infer emotion from the diagram. '
            'Override the one-label output format above: reply only JSON with exactly keys emotion and evidence. '
            'Copy classifier_label verbatim into emotion; put the requested visual comparison label into evidence.')
        return [{'role':'user','content':content}],images
    # Save one complete training triple for visual QA and exact pixel provenance.
    group=datasets['train'][0]['group']
    for record in [r for r in datasets['train'] if r['group']==group]:
        _,ims=messages(record,'actual')
        for j,im in enumerate(ims):im.save(OUT/f"training_example_{record['op']}_{j}.png")
    def inputs(records,answer=False,condition='actual'):
        texts=[];images=[];prompts=[]
        for r in records:
            ms,ims=messages(r,condition)
            prompt=processor.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompts.append(prompt)
            full=processor.apply_chat_template(ms+[{'role':'assistant','content':json.dumps({'emotion':predictions[r['index']]['label'],'evidence':r['target']})}],tokenize=False,add_generation_prompt=False,enable_thinking=False) if answer else prompt
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
            output=lm.generate(**batch,do_sample=False,max_new_tokens=96)
            texts=tok.batch_decode(output[:,n:],skip_special_tokens=True)
            for r,text in zip(rr,texts):
                copied=False
                try:
                    parsed=json.loads(text.strip())
                    pred=str(parsed.get('evidence','')).strip().upper()
                    copied=parsed.get('emotion')==predictions[r['index']]['label']
                except (ValueError,AttributeError):
                    pred=''
                supplied=predictions[r['index']]['label']
                rows.append({**r,'answer':text,'prediction':pred if pred in LABELS[r['task']] else None,'copied_class':copied,'classifier_label':supplied,'final_program_class':supplied})
        result={}
        for task,labs in LABELS.items():
            group=[r for r in rows if r['task']==task]
            recalls={lab:sum(r['target']==lab and r['prediction']==lab for r in group)/sum(r['target']==lab for r in group) for lab in labs}
            result[task]={'n':len(group),'bacc':sum(recalls.values())/3,'recall':recalls,
                'prediction_counts':{lab:sum(r['prediction']==lab for r in group) for lab in labs},'invalid':sum(r['prediction'] is None for r in group)}
        result['classifier_copy_accuracy']=sum(row['copied_class'] for row in rows)/len(rows)
        result['macro_bacc']=sum(result[t]['bacc'] for t in LABELS)/2
        grouped={}
        for row in rows:grouped.setdefault(row['group'],[]).append(row)
        result['paired']={}
        for task in LABELS:
            groups=[v for v in grouped.values() if v[0]['task']==task]
            assert all(len(g)==3 for g in groups)
            correct=lambda r:r['prediction']==r['target']
            result['paired'][task]={'groups':len(groups),
                'all_three_correct':sum(all(correct(r) for r in g) for g in groups)/len(groups),
                'both_directional_correct':sum(all(correct(r) for r in g if r['target']!='SIMILAR') for g in groups)/len(groups),
                'similar_correct':sum(correct(r) for g in groups for r in g if r['target']=='SIMILAR')/len(groups)}
        if condition=='actual':result['ce']=totalce/len(records)
        with (OUT/f'{tag}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return result
    if REEVALUATE:
        save('status.json',{'phase':'initial_validation_reevaluation'})
        initial_metrics=evaluate('initial_validation')
        available=[]
        for filename in ['best.pt','last.pt']:
            saved=torch.load(ORIGINAL_OUTPUT/filename,map_location='cpu',weights_only=False)
            set_peft_model_state_dict(lm,saved['lora'])
            with torch.no_grad():
                for name,parameter in lm.named_parameters():
                    if name in saved['merger']:parameter.copy_(saved['merger'][name].to(parameter.device))
            save('status.json',{'phase':'checkpoint_reevaluation','epoch':saved['epoch']})
            measured=evaluate(f"epoch_{saved['epoch']:02d}")
            available.append({'checkpoint':filename,'epoch':saved['epoch'],'validation':measured})
            save('available_checkpoint_metrics.json',available)
            del saved
        chosen=max(available,key=lambda item:(item['validation']['macro_bacc'],-item['validation']['ce']))
        saved=torch.load(ORIGINAL_OUTPUT/chosen['checkpoint'],map_location='cpu',weights_only=False)
        set_peft_model_state_dict(lm,saved['lora'])
        with torch.no_grad():
            for name,parameter in lm.named_parameters():
                if name in saved['merger']:parameter.copy_(saved['merger'][name].to(parameter.device))
        fixed=evaluate('selected_fixed','fixed')
        mismatch=evaluate('selected_mismatch','mismatch')
        save('complete.json',{'phase':'complete','initial':initial_metrics,'available':available,'selected':chosen,'fixed':fixed,'mismatch':mismatch,'max_new_tokens':96,'optimizer_steps':0,'test_read':False,'seconds':time.time()-start,'limitation':'Epoch2 weights were not retained. Selection is restricted to saved epochs1 and3; original zero scores and selection are invalid due to truncated JSON.'})
        save('status.json',{'phase':'complete'})
        return
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
        torch.save(ck,OUT/f'epoch_{ep:02d}.pt')
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

