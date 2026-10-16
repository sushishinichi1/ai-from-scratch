"""Evaluate Base checkpoints on the frozen Japanese common evaluation set."""
import argparse, json, math, re, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from generate import generate, load_model

ROOT=Path(__file__).resolve().parent
EVAL=ROOT/'datasets/japanese_base_common_eval'
PROMPTS=['人工知能の未来は','コンピュータとは','日本の歴史において','科学の発展によって','音楽は人間にとって']
SEEDS=[42]

def build_windows():
 tokens=np.load(EVAL/'tokens.npy',mmap_mode='r',allow_pickle=False)
 offsets=np.load(EVAL/'offsets.npy',allow_pickle=False)
 windows=[]
 for i in range(len(offsets)-1):
  a,b=int(offsets[i]),int(offsets[i+1]);n=b-a
  if n<2:continue
  doc=tokens[a:b]
  for start in range(0,n-1,128):
   end=min(start+128,n-1)
   windows.append((np.asarray(doc[start:end],dtype=np.int64),np.asarray(doc[start+1:end+1],dtype=np.int64)))
 return windows

@torch.inference_mode()
def common_loss(model,windows,batch_size=16):
 model.eval();total=0.0;count=0;device=next(model.parameters()).device
 for ix in range(0,len(windows),batch_size):
  chunk=windows[ix:ix+batch_size];width=max(len(x) for x,_ in chunk)
  x=torch.zeros((len(chunk),width),dtype=torch.long)
  y=torch.full((len(chunk),width),-100,dtype=torch.long)
  for j,(xx,yy) in enumerate(chunk):
   x[j,:len(xx)]=torch.from_numpy(xx);y[j,:len(yy)]=torch.from_numpy(yy)
  logits=model(x.to(device))
  summed=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.to(device).reshape(-1),ignore_index=-100,reduction='sum')
  total+=summed.item();count+=(y!=-100).sum().item()
 return total/count,count

def repetition(text,n):
 grams=[text[i:i+n] for i in range(max(0,len(text)-n+1))]
 if not grams:return 0.0
 counts=Counter(grams)
 return sum(v-1 for v in counts.values() if v>1)/len(grams)

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['v2','v3'],required=True);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 root=ROOT/f'checkpoints/japanese_base_{a.family}'
 if a.family=='v2':
  d=root/'wide30m_lr0003_bpe1024';items=[('3000',d/'update_3000.pt'),('10000',d/'update_10000.pt'),('18000',d/'update_18000.pt'),('best',d/'best.pt')]
  # The historical 300-update trial checkpoint is in the earlier 30M directory.
  items.insert(0,('300',ROOT/'checkpoints/japanese_base_v2/wide30m_bpe1024/update_300.pt'))
  out=ROOT/'logs/japanese_base_common_eval/v2'
 else:
  d=root/'wide30m_bpe1024_lr0003_common_eval'
  items=[(name,d/f'update_{step}.pt') for name,step in [('3000',3000),('10000',10000),('18000',18000)]]+ [('best',d/'best.pt')]
  out=ROOT/'logs/japanese_base_common_eval/v3'
 missing=[str(path) for _,path in items if not path.is_file()]
 if missing:raise FileNotFoundError('\n'.join(missing))
 if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing to overwrite {out}')
 out.mkdir(parents=True,exist_ok=True)
 windows=build_windows();result={'family':a.family,'evaluation_documents':900,'prompts':PROMPTS,
  'generation':{'temperature':0.8,'top_k':40,'max_new_tokens':300,'seed':42,'utf8_safe':True},
  'evaluation_loss_definition':'full document-level next-token cross entropy over all cached common-eval token transitions; document boundaries are not crossed',
  'checkpoints':[]}
 for label,path in items:
  start=time.perf_counter();model,tok=load_model(path,'cuda')
  if model.config.block_size!=128 or model.config.vocab_size!=1024:raise ValueError('Unexpected model config')
  loss,tokens=common_loss(model,windows)
  entry={'label':label,'checkpoint':str(path),'common_eval_loss':loss,'evaluated_target_tokens':tokens,'generations':[]}
  print(f'{a.family} {label}: common eval loss {loss:.6f} ({tokens:,} targets)',flush=True)
  for prompt in PROMPTS:
   for seed in SEEDS:
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    full=generate(model,tok,prompt,max_new_tokens=300,temperature=0.8,top_k=40)
    continuation=full[len(prompt):] if full.startswith(prompt) else full
    row={'prompt':prompt,'seed':seed,'text':full,'continuation':continuation,
      'generated_characters':len(continuation),'3gram_repetition':repetition(continuation,3),
      '4gram_repetition':repetition(continuation,4)}
    entry['generations'].append(row)
  entry['elapsed_seconds']=time.perf_counter()-start
  result['checkpoints'].append(entry)
  (out/f'{label}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf8')
  del model;torch.cuda.empty_cache()
 result['mean_repetition']={label:{'3gram':float(np.mean([g['3gram_repetition'] for g in c['generations']])),'4gram':float(np.mean([g['4gram_repetition'] for g in c['generations']]))} for label,c in ((x['label'],x) for x in result['checkpoints'])}
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'family':a.family,'common_eval':[(x['label'],x['common_eval_loss']) for x in result['checkpoints']],'mean_repetition':result['mean_repetition'],'output':str(out)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
