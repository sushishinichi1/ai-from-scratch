"""Fresh assistant-only SFT on the isolated Japanese 5000-example dataset."""
import json, math, time, copy, re
from pathlib import Path
import torch
from torch.nn import functional as F
from bpe_tokenizer import BPETokenizer
from chat_data import format_prompt, load_instruction_examples, collate_instruction_batch
from generate import generate
from model import ModelConfig,TinyGPT

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'checkpoints/japanese_base_v2/wide30m_lr0003_bpe1024/best.pt'
CHAT1K=ROOT/'checkpoints/japanese_base_v2/chat30m_1000/lr0001/best.pt'
DATA=ROOT/'datasets/japanese_base_v2/chat_5000'
OUT=ROOT/'checkpoints/japanese_base_v2/chat30m_5000_lr0001'
LOG=ROOT/'logs/japanese_base_v2/chat30m_long_run'
TOKPATH=ROOT/'datasets/japanese_base_v2/bpe_1024.json'
SEED=42; BATCH=8; LR=1e-4; CLIP=1.
FIXED=json.loads((DATA/'fixed_test_questions.json').read_text(encoding='utf-8'))

def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)
def loss_fn(model,batch,reduce='mean'):
    x,y,m=batch; logits=model(x); vals=F.cross_entropy(logits.reshape(-1,model.config.vocab_size),y.reshape(-1),reduction='none').view_as(y).masked_select(m)
    if not vals.numel():raise ValueError('no assistant targets')
    return vals.sum() if reduce=='sum' else vals.mean()
def evaluate(model,data):
    was=model.training;model.eval();tot=0.;n=0
    with torch.inference_mode():
        for s in range(0,len(data),BATCH):
            b=collate_instruction_batch(data,list(range(s,min(s+BATCH,len(data)))),'cuda');tot+=loss_fn(model,b,'sum').item();n+=int(b[2].sum())
    model.train(was);v=tot/n
    if not math.isfinite(v):raise FloatingPointError('nonfinite eval loss')
    return v
def ngram_rate(s,n):
    x=[c for c in s if not c.isspace()];g=[tuple(x[i:i+n]) for i in range(max(0,len(x)-n+1))]
    return (len(g)-len(set(g)))/len(g) if g else 0.
def generate_row(model,tok,q,seed=42):
    prompt=format_prompt(q);prompt_ids=tok.encode(prompt)
    if len(prompt_ids)>128 or tok.decode(prompt_ids)!=prompt:raise ValueError('bad prompt serialization')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model.eval()
    with torch.inference_mode():full=generate(model,tok,prompt,max_new_tokens=120,temperature=.7,top_k=40)
    if not full.startswith(prompt):raise ValueError('generated prompt prefix mismatch')
    raw=full[len(prompt):];answer=raw.split('<|user|>',1)[0].split('<|assistant|>',1)[0].strip()
    ascii_letters=sum(c.isascii() and c.isalpha() for c in answer);letters=sum(c.isalpha() for c in answer)
    return {'question':q,'prompt':prompt,'answer':answer,'raw_continuation':raw,'utf8_safe':'\ufffd' not in full,
            'character_count':len(answer),'3gram_repetition_rate':ngram_rate(answer,3),'4gram_repetition_rate':ngram_rate(answer,4),
            'empty':not bool(answer),'english_or_latin_ratio':ascii_letters/max(1,letters)}
def gen_test(model,tok,questions=FIXED):return [generate_row(model,tok,q) for q in questions]
def checkpoint(path,model,opt,tok,cfg,step,batch_rng,history,best):
    p={'model_config':cfg.to_dict(),'model_state_dict':cpu_tree(model.state_dict()),'optimizer_state_dict':cpu_tree(opt.state_dict()),
       'tokenizer_type':'bpe','tokenizer_config':tok.to_dict(),'step':step,'base_checkpoint':str(BASE),
       'train_path':str(DATA/'train.jsonl'),'validation_path':str(DATA/'validation.jsonl'),
       'training_config':{'lr':LR,'batch_size':BATCH,'seed':SEED,'dtype':'float32','gradient_clipping':CLIP,'objective':'assistant-answer-only','context':128,'fresh_from_base':True},
       'cpu_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all(),'batch_generator_state':batch_rng.get_state(),
       'loss_history':history,'best_validation_loss':best}
    tmp=path.with_suffix('.pt.tmp');torch.save(p,tmp);tmp.replace(path)
def human_flags(rows):
    # Explicit proxies to help review; natural-answer count is manually represented
    # by a conservative automatic screen and the raw completions remain in JSON.
    for x in rows:
        x['english_or_corrupt_or_empty']=x['empty'] or not x['utf8_safe'] or x['english_or_latin_ratio']>.4
        x['repetitive']=x['3gram_repetition_rate']>.45 or x['4gram_repetition_rate']>.35
        # A conservative Japanese relevance proxy: share nontrivial character bigrams.
        q=re.sub(r'[^一-龠ぁ-んァ-ヶA-Za-z0-9]','',x['question']);a=re.sub(r'[^一-龠ぁ-んァ-ヶA-Za-z0-9]','',x['answer'])
        qg={q[i:i+2] for i in range(max(0,len(q)-1))};ag={a[i:i+2] for i in range(max(0,len(a)-1))}
        x['question_overlap_ratio']=len(qg&ag)/max(1,len(qg))
        x['auto_natural_answer_proxy']=bool(x['answer']) and not x['english_or_corrupt_or_empty'] and not x['repetitive'] and x['question_overlap_ratio']>=.08
    return rows
def load_checkpoint_model(path,device='cuda'):
    c=torch.load(path,map_location='cpu',weights_only=False);cfg=ModelConfig(**c['model_config']);m=TinyGPT(cfg);m.load_state_dict(c['model_state_dict']);m.to(device,dtype=torch.float32).eval();return m,c
def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'refusing output {OUT}')
    if LOG.exists() and (LOG/'phase4_phase5_5000.json').exists():raise FileExistsError('refusing existing result')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    base=torch.load(BASE,map_location='cpu',weights_only=True);cfg=ModelConfig(**base['model_config']);tok=BPETokenizer.from_dict(base['tokenizer_config'])
    filetok=BPETokenizer.load(TOKPATH)
    if tok.to_dict()!=filetok.to_dict() or tok.vocab_size!=1024:raise ValueError('tokenizer mismatch')
    if (cfg.vocab_size,cfg.embedding_dim,cfg.num_heads,cfg.num_layers,cfg.block_size,cfg.attention_impl)!=(1024,640,8,6,128,'custom'):raise ValueError('unexpected Base configuration')
    tr=load_instruction_examples(DATA/'train.jsonl',tok,128);va=load_instruction_examples(DATA/'validation.jsonl',tok,128);te=load_instruction_examples(DATA/'test.jsonl',tok,128)
    if (len(tr),len(va),len(te))!=(4000,500,500):raise ValueError('dataset counts mismatch')
    model=TinyGPT(cfg);model.load_state_dict(base['model_state_dict']);del base;model.to('cuda',dtype=torch.float32).train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);brng=torch.Generator().manual_seed(SEED)
    OUT.mkdir(parents=True);LOG.mkdir(parents=True,exist_ok=True)
    hist=[{'update':0,'train_loss':evaluate(model,tr),'validation_loss':evaluate(model,va)}]
    best=hist[0]['validation_loss'];beststep=0;bestmodel=cpu_tree(model.state_dict());bestopt=cpu_tree(opt.state_dict())
    bestpath=OUT/'best.pt';checkpoint(bestpath,model,opt,tok,cfg,0,brng,hist,best)
    evals={50,100,150,200,250,300};gens={};torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
    maxstep=300
    for step in range(1,maxstep+1):
        ix=torch.randint(len(tr),(BATCH,),generator=brng).tolist();b=collate_instruction_batch(tr,ix,'cuda')
        opt.zero_grad(set_to_none=True);loss=loss_fn(model,b)
        if not math.isfinite(loss.item()):raise FloatingPointError(f'nonfinite train loss {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP,error_if_nonfinite=True);opt.step()
        if step in evals:
            rec={'update':step,'train_loss':evaluate(model,tr),'validation_loss':evaluate(model,va)};hist.append(rec)
            if rec['validation_loss']<best:
                best=rec['validation_loss'];beststep=step;bestmodel=cpu_tree(model.state_dict());bestopt=cpu_tree(opt.state_dict());checkpoint(bestpath,model,opt,tok,cfg,step,brng,hist,best)
            print(f'5000data update={step} train={rec["train_loss"]:.6f} val={rec["validation_loss"]:.6f} best={best:.6f}@{beststep}',flush=True)
            if step in (100,200,300):
                gens[str(step)]=human_flags(gen_test(model,tok));model.train()
    # Continue to 1000 only if 300 is improving relative to both preceding checkpoints.
    if hist[-1]['validation_loss']<hist[-2]['validation_loss'] and hist[-2]['validation_loss']<hist[-3]['validation_loss']:
        maxstep=1000
        for step in range(301,1001):
            ix=torch.randint(len(tr),(BATCH,),generator=brng).tolist();b=collate_instruction_batch(tr,ix,'cuda')
            opt.zero_grad(set_to_none=True);loss=loss_fn(model,b)
            if not math.isfinite(loss.item()):raise FloatingPointError(f'nonfinite train loss {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP,error_if_nonfinite=True);opt.step()
            if step in (400,500,600,800,1000):
                rec={'update':step,'train_loss':evaluate(model,tr),'validation_loss':evaluate(model,va)};hist.append(rec)
                if rec['validation_loss']<best:
                    best=rec['validation_loss'];beststep=step;bestmodel=cpu_tree(model.state_dict());bestopt=cpu_tree(opt.state_dict());checkpoint(bestpath,model,opt,tok,cfg,step,brng,hist,best)
                print(f'5000data update={step} train={rec["train_loss"]:.6f} val={rec["validation_loss"]:.6f} best={best:.6f}@{beststep}',flush=True)
                # Stop only after sustained, material val deterioration after the best.
                recent=[x['validation_loss'] for x in hist[-3:]]
                if len(recent)==3 and all(v>best+.08 for v in recent):
                    print('early stop: sustained validation deterioration',flush=True);break
    torch.cuda.synchronize();elapsed=time.perf_counter()-t0
    # Read back the selected best checkpoint for final comparison.
    bestmodelnet,chat5=load_checkpoint_model(bestpath)
    # Generate all fixed prompts from Base, 1000-example best, and 5000-example best.
    base_model,base_c=load_checkpoint_model(BASE);chat1_model,c1=load_checkpoint_model(CHAT1K)
    compare={'settings':{'temperature':.7,'top_k':40,'max_new_tokens':120,'seed':42,'utf8_safe':True},'questions':FIXED,'models':{}}
    for label,m in [('base',base_model),('1000_best',chat1_model),('5000_best',bestmodelnet)]:
        rows=human_flags(gen_test(m,tok));compare['models'][label]={'checkpoint':str({'base':BASE,'1000_best':CHAT1K,'5000_best':bestpath}[label]),'generations':rows,
            'auto_natural_answer_count':sum(x['auto_natural_answer_proxy'] for x in rows),'mean_3gram_repetition_rate':sum(x['3gram_repetition_rate'] for x in rows)/len(rows),
            'mean_4gram_repetition_rate':sum(x['4gram_repetition_rate'] for x in rows)/len(rows)}
    # Persist all 60 final comparative generations plus intermediate 5k snapshots.
    (LOG/'fixed_test20_comparison.json').write_text(json.dumps(compare,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'phase4_phase5_5000.json').write_text(json.dumps({'dataset_report':json.loads((DATA/'dataset_report.json').read_text(encoding='utf-8')),
        'history':hist,'best_update':beststep,'best_validation_loss':best,'training_seconds':elapsed,
        'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'checkpoint':str(bestpath),'generation_snapshots':gens,'comparison_json':str(LOG/'fixed_test20_comparison.json'),
        'comparison_summary':{k:{'auto_natural_answer_count':v['auto_natural_answer_count'],'mean_3gram_repetition_rate':v['mean_3gram_repetition_rate'],'mean_4gram_repetition_rate':v['mean_4gram_repetition_rate']} for k,v in compare['models'].items()}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"5000 RESULT best={best:.6f}@{beststep}; time={elapsed:.1f}s; VRAM={torch.cuda.max_memory_allocated()/2**20:.1f}MiB",flush=True)
    for k,v in compare['models'].items():print(f"{k}: auto_natural={v['auto_natural_answer_count']}/20 rep3={v['mean_3gram_repetition_rate']:.3f} rep4={v['mean_4gram_repetition_rate']:.3f}",flush=True)
if __name__=='__main__':main()
