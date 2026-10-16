"""Resume v3 update 300 into an isolated branch through update 18000."""
import json, math, time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import get_batch
from evaluate_japanese_base_common import build_windows, common_loss
from model import ModelConfig30M, TinyGPT
from train_shakespeare import estimate_loss, tokenizer_state

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/japanese_base_v3'
TOKENIZER_PATH=ROOT/'datasets/japanese_base_v2/bpe_1024.json'
RESUME=ROOT/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003/update_300.pt'
OUT=ROOT/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_common_eval'
LOG=ROOT/'logs/japanese_base_v3/common_eval_18000'
SEED=42;LR=.0003;BATCH=16;CLIP=1.0;TOTAL=18000
EVAL_STEPS=(1000,3000,5000,7500,10000,12500,15000,18000)
SAVE_STEPS=(3000,5000,10000,15000,18000)

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'refusing to overwrite {OUT}')
 if LOG.exists() and any(LOG.iterdir()):raise FileExistsError(f'refusing to overwrite {LOG}')
 device='cuda'
 tok=BPETokenizer.load(TOKENIZER_PATH)
 train=torch.tensor(np.load(DATA/'train_tokens.npy',allow_pickle=False),dtype=torch.long)
 val=torch.tensor(np.load(DATA/'validation_tokens.npy',allow_pickle=False),dtype=torch.long)
 common_windows=build_windows()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest');torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 cfg=ModelConfig30M(vocab_size=1024,block_size=128)
 model=TinyGPT(cfg).to(device=device,dtype=torch.float32)
 if sum(p.numel() for p in model.parameters())!=30_936_064:raise ValueError('parameter mismatch')
 optimizer=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 state=torch.load(RESUME,map_location='cpu',weights_only=True)
 if state.get('step')!=300:raise ValueError('resume checkpoint is not update 300')
 if state.get('model_config')!=cfg.to_dict():raise ValueError('resume model config mismatch')
 if not state.get('optimizer_state_dict',{}).get('state'):raise ValueError('optimizer state missing; cannot faithfully resume')
 for key in ('cpu_rng_state','cuda_rng_states','batch_generator_state'):
  if key not in state:raise ValueError(f'missing required RNG state: {key}')
 if state.get('training_config',{}).get('lr',LR)!=LR:raise ValueError('resume learning-rate mismatch')
 if state.get('training_config',{}).get('batch_size',BATCH)!=BATCH:raise ValueError('resume batch-size mismatch')
 model.load_state_dict(state['model_state_dict'])
 optimizer.load_state_dict(state['optimizer_state_dict'])
 batch_rng=torch.Generator();batch_rng.set_state(state['batch_generator_state'])
 torch.set_rng_state(state['cpu_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_states'])
 history=list(state.get('loss_history',[]));best_val=float(state['best_val_loss']);best_update=int(state['best_update'])
 run={'resume_checkpoint':str(RESUME),'resumed_step':300,'optimizer_restored':True,
  'cpu_rng_restored':True,'cuda_rng_restored':True,'batch_rng_restored':True,
  'model_config':cfg.to_dict(),'parameters':30_936_064,'batch_size':BATCH,
  'effective_tokens_per_update':BATCH*128,'lr':LR,'seed':SEED,'precision':'FP32',
  'gradient_clipping':CLIP,'optimizer':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,
  'common_eval_never_used_for_training_or_checkpoint_selection':True,
  'common_eval_windows':len(common_windows)}
 OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
 run_id=time.time_ns();log_path=LOG/f'continued_300to18000_{run_id}.json'
 metrics={'config':run,'loss_history':history,'updates':300,'best_update':best_update,'best_validation_loss':best_val}
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 def save(path,step):
  tt,tc=tokenizer_state(tok)
  cp={'model_config':cfg.to_dict(),'model_state_dict':model.state_dict(),
   'optimizer_state_dict':optimizer.state_dict(),'tokenizer_type':tt,'tokenizer_config':tc,
   'step':step,'best_update':best_update,'best_val_loss':best_val,
   'cpu_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all(),
   'batch_generator_state':batch_rng.get_state(),'run_config':run,'loss_history':list(history),
   'resume_checkpoint':str(RESUME)}
  temp=path.with_suffix('.pt.tmp');torch.save(cp,temp);temp.replace(path)
 print(f"Resume update 300; GPU={torch.cuda.get_device_name(0)}; common eval targets="
       f"{sum(len(y) for _,y in common_windows):,}; best Val={best_val:.6f}",flush=True)
 model.train()
 for step in range(301,TOTAL+1):
  x,y=get_batch(train,128,BATCH,device,batch_rng)
  optimizer.zero_grad(set_to_none=True);logits=model(x)
  loss=F.cross_entropy(logits.reshape(-1,1024),y.reshape(-1))
  if not math.isfinite(loss.item()):raise FloatingPointError(f'nonfinite loss update {step}')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP,error_if_nonfinite=True);optimizer.step()
  if step in EVAL_STEPS:
   tr=estimate_loss(model,train,device,batches=10);vl=estimate_loss(model,val,device,batches=10)
   ce,ct=common_loss(model,common_windows)
   row={'update':step,'train_loss':tr,'validation_loss':vl,'common_eval_loss':ce,'common_eval_targets':ct}
   history.append(row)
   if vl<best_val:best_val=vl;best_update=step;save(OUT/'best.pt',step)
   torch.cuda.synchronize();elapsed=time.perf_counter()-started
   metrics.update({'updates':step,'best_update':best_update,'best_validation_loss':best_val,
    'elapsed_seconds':elapsed,'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
    'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
   log_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
   print(f"Update {step}: Train={tr:.6f}; Val={vl:.6f}; Common={ce:.6f}; "
         f"best={best_val:.6f}@{best_update}; elapsed={elapsed/60:.1f}m",flush=True)
   if step in SAVE_STEPS:save(OUT/f'update_{step}.pt',step)
 torch.cuda.synchronize();elapsed=time.perf_counter()-started
 metrics.update({'updates':TOTAL,'best_update':best_update,'best_validation_loss':best_val,
  'elapsed_seconds':elapsed,'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
  'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
  'checkpoint':str(OUT/f'update_{TOTAL}.pt'),'best_checkpoint':str(OUT/'best.pt')})
 log_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
 print(f"Completed; elapsed={elapsed/3600:.2f}h; peak VRAM={metrics['peak_vram_allocated_mib']:.1f} MiB; "
       f"best={best_val:.6f}@{best_update}",flush=True)
if __name__=='__main__':main()
