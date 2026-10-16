"""Resume Japanese Base v3 from update 18000 through update 48000, then evaluate generations."""
import json, math, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import get_batch
from evaluate_japanese_base_common import build_windows, common_loss
from generate import generate, load_model
from model import ModelConfig30M, TinyGPT
from train_shakespeare import estimate_loss, tokenizer_state

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/japanese_base_v3'
TOKENIZER_PATH=ROOT/'datasets/japanese_base_v2/bpe_1024.json'
RESUME=ROOT/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_common_eval/update_18000.pt'
OUT=ROOT/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_run18000to48000'
LOG=ROOT/'logs/japanese_base_v3/run18000to48000'
SEED=42;LR=.0003;BATCH=16;CLIP=1.0;TOTAL=48000;START=18000
EVAL_STEPS=(20000,24000,28000,32000,36000,40000,44000,48000)
SAVE_STEPS=(24000,32000,40000,48000)
GEN_STEPS=(24000,32000,40000,48000,'best')
PROMPTS=['人工知能の未来は','コンピュータとは','日本の歴史において','科学の発展によって','音楽は人間にとって']

def repetition(text,n):
 grams=[text[i:i+n] for i in range(max(0,len(text)-n+1))]
 if not grams:return 0.0
 counts=Counter(grams)
 return sum(v-1 for v in counts.values() if v>1)/len(grams)

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'refusing to overwrite {OUT}')
 if LOG.exists() and any(LOG.iterdir()):raise FileExistsError(f'refusing to overwrite {LOG}')
 tok=BPETokenizer.load(TOKENIZER_PATH)
 train=torch.tensor(np.load(DATA/'train_tokens.npy',allow_pickle=False),dtype=torch.long)
 val=torch.tensor(np.load(DATA/'validation_tokens.npy',allow_pickle=False),dtype=torch.long)
 common_windows=build_windows()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest');torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 cfg=ModelConfig30M(vocab_size=1024,block_size=128)
 model=TinyGPT(cfg).to(device='cuda',dtype=torch.float32)
 if sum(p.numel() for p in model.parameters())!=30_936_064:raise ValueError('parameter mismatch')
 optimizer=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 cp=torch.load(RESUME,map_location='cpu',weights_only=True)
 if cp.get('step')!=START or cp.get('model_config')!=cfg.to_dict():raise ValueError('resume checkpoint config/step mismatch')
 if not cp.get('optimizer_state_dict',{}).get('state'):raise ValueError('optimizer state absent')
 for key in ('cpu_rng_state','cuda_rng_states','batch_generator_state'):
  if key not in cp:raise ValueError(f'missing RNG state {key}')
 oldrun=cp.get('run_config',{})
 for key,want in [('lr',LR),('seed',SEED),('batch_size',BATCH),('gradient_clipping',CLIP)]:
  if oldrun.get(key,want)!=want:raise ValueError(f'resume config mismatch: {key}')
 model.load_state_dict(cp['model_state_dict']);optimizer.load_state_dict(cp['optimizer_state_dict'])
 batch_rng=torch.Generator();batch_rng.set_state(cp['batch_generator_state'])
 torch.set_rng_state(cp['cpu_rng_state']);torch.cuda.set_rng_state_all(cp['cuda_rng_states'])
 history=list(cp.get('loss_history',[]));best_val=float(cp['best_val_loss']);best_update=int(cp['best_update'])
 run={'resume_checkpoint':str(RESUME),'resume_update':START,'optimizer_restored':True,
  'cpu_rng_restored':True,'cuda_rng_restored':True,'batch_rng_restored':True,
  'model_config':cfg.to_dict(),'parameters':30_936_064,'batch_size':BATCH,
  'effective_tokens_per_update':BATCH*128,'lr':LR,'seed':SEED,'precision':'FP32',
  'gradient_clipping':CLIP,'optimizer':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,
  'common_eval_only_for_reporting':True,'best_checkpoint_selection':'v3 validation loss only',
  'common_eval_target_tokens':sum(len(y) for _,y in common_windows)}
 OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
 run_id=time.time_ns();log_path=LOG/f'continued_18000to48000_{run_id}.json'
 metrics={'config':run,'loss_history':history,'updates':START,'best_update':best_update,'best_validation_loss':best_val}
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 def save(path,step):
  tt,tc=tokenizer_state(tok)
  state={'model_config':cfg.to_dict(),'model_state_dict':model.state_dict(),
   'optimizer_state_dict':optimizer.state_dict(),'tokenizer_type':tt,'tokenizer_config':tc,
   'step':step,'best_update':best_update,'best_val_loss':best_val,
   'cpu_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all(),
   'batch_generator_state':batch_rng.get_state(),'run_config':run,
   'loss_history':list(history),'resume_checkpoint':str(RESUME)}
  temp=path.with_suffix('.pt.tmp');torch.save(state,temp);temp.replace(path)
 # Seed the new branch's best with its resumed baseline; later replacements
 # require an actual v3-validation improvement.
 save(OUT/'best.pt',START)
 print(f'Resuming from {START}; GPU={torch.cuda.get_device_name(0)}; '
       f'common targets={run["common_eval_target_tokens"]:,}',flush=True)
 model.train()
 for step in range(START+1,TOTAL+1):
  x,y=get_batch(train,128,BATCH,'cuda',batch_rng)
  optimizer.zero_grad(set_to_none=True);logits=model(x)
  loss=F.cross_entropy(logits.reshape(-1,1024),y.reshape(-1))
  if not math.isfinite(loss.item()):raise FloatingPointError(f'nonfinite train loss at {step}')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP,error_if_nonfinite=True);optimizer.step()
  if step in EVAL_STEPS:
   tr=estimate_loss(model,train,'cuda',batches=10);vl=estimate_loss(model,val,'cuda',batches=10)
   ce,ct=common_loss(model,common_windows)
   row={'update':step,'train_loss':tr,'validation_loss':vl,'common_eval_loss':ce,'common_eval_targets':ct}
   history.append(row)
   if vl<best_val:best_val=vl;best_update=step;save(OUT/'best.pt',step)
   torch.cuda.synchronize();elapsed=time.perf_counter()-started
   metrics.update({'updates':step,'best_update':best_update,'best_validation_loss':best_val,
    'elapsed_seconds':elapsed,'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
    'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
   log_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
   print(f'Update {step}: Train={tr:.6f}; Val={vl:.6f}; Common={ce:.6f}; '
         f'best={best_val:.6f}@{best_update}; elapsed={elapsed/60:.1f}m',flush=True)
   if step in SAVE_STEPS:save(OUT/f'update_{step}.pt',step)
 torch.cuda.synchronize();elapsed=time.perf_counter()-started
 metrics.update({'updates':TOTAL,'best_update':best_update,'best_validation_loss':best_val,
  'elapsed_seconds':elapsed,'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
  'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
  'checkpoint':str(OUT/'update_48000.pt'),'best_checkpoint':str(OUT/'best.pt')})
 log_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
 print(f'Training complete: {elapsed/3600:.2f}h; best={best_val:.6f}@{best_update}; '
       f'peak VRAM={metrics["peak_vram_allocated_mib"]:.1f} MiB',flush=True)
 del model,optimizer,logits,loss,x,y;torch.cuda.empty_cache()
 generations=[]
 for label in GEN_STEPS:
  path=OUT/'best.pt' if label=='best' else OUT/f'update_{label}.pt'
  gm,gt=load_model(path,'cuda');items=[]
  for prompt in PROMPTS:
   torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   full=generate(gm,gt,prompt,max_new_tokens=300,temperature=.8,top_k=40)
   continuation=full[len(prompt):] if full.startswith(prompt) else full
   items.append({'prompt':prompt,'text':full,'continuation':continuation,
    'generated_characters':len(continuation),'3gram_repetition':repetition(continuation,3),
    '4gram_repetition':repetition(continuation,4)})
  generations.append({'label':label,'checkpoint':str(path),'generations':items,
   'mean_3gram_repetition':float(np.mean([g['3gram_repetition'] for g in items])),
   'mean_4gram_repetition':float(np.mean([g['4gram_repetition'] for g in items]))})
  del gm;torch.cuda.empty_cache()
  print(f'Generated checkpoint {label}',flush=True)
 previous=json.loads((ROOT/'logs/japanese_base_common_eval/v3/summary.json').read_text(encoding='utf8'))
 baseline=next(c for c in previous['checkpoints'] if c['label']=='18000')
 generations.insert(0,{'label':18000,'checkpoint':baseline['checkpoint'],'generations':baseline['generations'],
  'mean_3gram_repetition':previous['mean_repetition']['18000']['3gram'],
  'mean_4gram_repetition':previous['mean_repetition']['18000']['4gram']})
 generation_report={'conditions':{'temperature':.8,'top_k':40,'max_new_tokens':300,'seed':42,'utf8_safe':True},
  'prompts':PROMPTS,'checkpoints':generations}
 (LOG/'generation_comparison.json').write_text(json.dumps(generation_report,ensure_ascii=False,indent=2),encoding='utf8')
 metrics['generation_report']=str(LOG/'generation_comparison.json')
 log_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
 print(f'All requested training and generation saved under {OUT} and {LOG}',flush=True)
if __name__=='__main__':main()
