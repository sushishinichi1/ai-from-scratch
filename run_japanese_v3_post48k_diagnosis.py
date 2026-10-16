import json, math, random, re, time, collections, hashlib
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from model import ModelConfig,TinyGPT
from bpe_tokenizer import BPETokenizer
from generate import generate,load_model
from evaluate_japanese_base_common import build_windows
from chat_data import format_prompt,load_instruction_examples,collate_instruction_batch
R=Path(__file__).resolve().parent; O=R/'logs/japanese_base_v3/post48k_diagnosis'; C=R/'checkpoints/japanese_base_v3/chat_transfer_comparison'
BASE={'18000':R/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_common_eval/update_18000.pt','32000':R/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_run18000to48000/update_32000.pt','48000':R/'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003_run18000to48000/update_48000.pt'}
TOK=R/'datasets/japanese_base_v2/bpe_1024.json'; V=R/'datasets/japanese_base_v3'; CE=R/'datasets/japanese_base_common_eval'; CHAT=R/'datasets/japanese_base_v2/chat_5000'
P=[('AI','人工知能が社会に広がると、私たちの暮らしは'),('AI','機械学習の仕組みを考えると'),('computer','コンピュータのCPUとメモリは'),('computer','インターネットで情報を安全に扱うには'),('science','科学者が仮説を検証するとき'),('science','植物が光を受けて成長する過程では'),('history','日本の近代化が進んだ背景には'),('history','歴史資料を読むときに大切なのは'),('society','地域社会で公共交通を維持するには'),('economy','物価が上昇すると家計には'),('nature','森林と海の生態系は'),('music','音楽を聴くことが人の気分に与える影響は'),('daily','休日に新しい料理を作るなら'),('explanation','再生可能エネルギーとは'),('explanation','水が蒸発して雲になるまでを説明すると')]
def save(name,x):
 q=O/name;t=q.with_suffix(q.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');t.replace(q)
def cp(path):return torch.load(path,map_location='cpu',weights_only=False)
def rep(s,n):
 a=[c for c in s if not c.isspace()];g=[tuple(a[i:i+n]) for i in range(max(0,len(a)-n+1))]
 return (len(g)-len(set(g)))/len(g) if g else 0
def loaded(path):
 x=cp(path);cfg=ModelConfig(**x['model_config']);m=TinyGPT(cfg);m.load_state_dict(x['model_state_dict']);m.to('cuda',dtype=torch.float32).eval();return m,x
def main():
 start=time.perf_counter()
 if not torch.cuda.is_available():raise RuntimeError('CUDA inaccessible; refusing CPU fallback')
 # Resume this same runner after the completed generation phase.
 for p in [*BASE.values(),TOK,CE/'manifest.json',CE/'tokens.npy',V/'manifest.json',V/'train_documents.jsonl',V/'validation_documents.jsonl',V/'train_tokens.npy',CHAT/'train.jsonl',CHAT/'validation.jsonl',CHAT/'test.jsonl',CHAT/'fixed_test_questions.json']:
  if not p.exists():raise FileNotFoundError(p)
 for k,p in BASE.items():
  if int(cp(p).get('step',-1))!=int(k):raise ValueError('step mismatch '+k)
 tok=BPETokenizer.load(TOK)
 if tok.vocab_size!=1024:raise ValueError('tokenizer vocab')
 O.mkdir(parents=True,exist_ok=True);C.mkdir(parents=True,exist_ok=True)
 cm=json.loads((CE/'manifest.json').read_text(encoding='utf8')); vm=json.loads((V/'manifest.json').read_text(encoding='utf8'))
 print('PREFLIGHT GPU',torch.cuda.get_device_name(0),'common docs/tokens',cm.get('documents'),cm.get('tokens'),'v3 train',vm['train'].get('tokens'),flush=True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
 out={'preflight':{'gpu':torch.cuda.get_device_name(0),'no_training_process':True,'checkpoints':{k:str(v) for k,v in BASE.items()},'common_eval_manifest':cm,'v3_manifest':vm,'tokenizer':str(TOK)},'prompts':[{'category':a,'text':b} for a,b in P]}
 if (O/'generation_matrix.json').exists():
  saved=json.loads((O/'generation_matrix.json').read_text(encoding='utf8'));out.update(saved);print('PHASE1 reused saved generation matrix',flush=True)
 else:
  # P1 controlled 15 prompt x 3 seeds x 3 decoding configs x 3 checkpoints
  gs=[]
  for lab,path in BASE.items():
   model,_=loaded(path)
   for cond,t,k in [('A',.8,40),('B',.6,40),('C',.8,20)]:
    for cat,prompt in P:
     for seed in (42,123,777):
      torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
      with torch.inference_mode():full=generate(model,tok,prompt,max_new_tokens=180,temperature=t,top_k=k)
      s=full[len(prompt):] if full.startswith(prompt) else full;ids=tok.encode(s);pids=set(tok.encode(prompt));words=re.findall(r'\S+',s)
      gs.append({'checkpoint':lab,'condition':cond,'temperature':t,'top_k':k,'category':cat,'prompt':prompt,'seed':seed,'text':full,'continuation':s,'characters':len(s),'tokens':len(ids),'3gram_repetition':rep(s,3),'4gram_repetition':rep(s,4),'unique_token_ratio':len(set(ids))/max(1,len(ids)),'consecutive_same_token_pairs':sum(x==y for x,y in zip(ids,ids[1:])),'consecutive_same_words':sum(x==y for x,y in zip(words,words[1:])),'prompt_token_coverage':len(pids&set(ids))/max(1,len(pids)),'empty':not bool(s.strip()),'replacement_char':'\ufffd' in full})
   del model;torch.cuda.empty_cache();print('generation checkpoint',lab,flush=True)
  out['generations']=gs;out['generation_summary']={}
  for lab in BASE:
   out['generation_summary'][lab]={}
   for cond in 'ABC':
    rows=[x for x in gs if x['checkpoint']==lab and x['condition']==cond]
    out['generation_summary'][lab][cond]={k:float(np.mean([x[k] for x in rows])) for k in ['3gram_repetition','4gram_repetition','unique_token_ratio','characters','prompt_token_coverage','consecutive_same_token_pairs']}
    out['generation_summary'][lab][cond]['empty_count']=sum(x['empty'] for x in rows);out['generation_summary'][lab][cond]['replacement_count']=sum(x['replacement_char'] for x in rows)
  save('generation_matrix.json',out);print('PHASE1 complete',flush=True)
 # P2 fixed shared common-eval subset next-token distribution
 wins=build_windows();ix=np.linspace(0,len(wins)-1,min(256,len(wins)),dtype=int);fixed=[wins[int(i)] for i in ix];dist={}
 for lab,path in BASE.items():
  m,_=loaded(path);sums=np.zeros(4,dtype=np.float64);n=0
  with torch.inference_mode():
   for i in range(len(fixed)):
    x=torch.as_tensor(np.asarray(fixed[i][0])).reshape(1,-1).cuda();y=torch.as_tensor(np.asarray(fixed[i][1])).reshape(1,-1).cuda()
    z=m(x).float();lp=F.log_softmax(z,-1);pr=lp.exp();sums[0]+=F.nll_loss(lp.flatten(0,1),y.flatten(),reduction='sum').item();sums[1]+=(-(pr*lp).sum(-1)).sum().item();sums[2]+=pr.max(-1).values.sum().item();sums[3]+=pr.topk(5,-1).values.sum(-1).sum().item();n+=y.numel()
  dist[lab]={'targets':n,'cross_entropy':sums[0]/n,'mean_entropy_nats':sums[1]/n,'mean_top1_probability':sums[2]/n,'mean_top5_mass':sums[3]/n,'nonfinite':not np.isfinite(sums).all()}
  del m;torch.cuda.empty_cache()
 out['next_token_distribution']=dist;save('next_token_distribution.json',out);print('PHASE2',dist,flush=True)
 # P3 source-stratified validation; deterministic sample, cap 160 windows/source
 groups=collections.defaultdict(list)
 for line in (V/'validation_documents.jsonl').open(encoding='utf8'):
  d=json.loads(line);src=d.get('source','unknown');src='wiki_family' if src in ('wikipedia','wikibooks','wikinews') else src
  txt=d.get('text',d.get('cleaned_text',''))
  if txt.strip():groups[src].append((str(d.get('id','')),txt))
 sd={}
 for src,rows in groups.items():
  rows=sorted(rows,key=lambda x:hashlib.sha1((src+x[0]).encode()).hexdigest())[:64];seq=[]
  for _,text in rows:
   ids=tok.encode(text)
   for j in range(0,len(ids)-128,128):
    if j+129<=len(ids):seq.append(ids[j:j+129])
    if len(seq)>=160:break
   if len(seq)>=160:break
  sd[src]=seq
 srcres={}
 for lab,path in BASE.items():
  m,_=loaded(path);srcres[lab]={}
  with torch.inference_mode():
   for src,seqs in sd.items():
    total=n=0
    for i in range(0,len(seqs),4):
     a=torch.tensor([q[:-1] for q in seqs[i:i+4]],device='cuda');b=torch.tensor([q[1:] for q in seqs[i:i+4]],device='cuda');z=m(a).float();total+=F.cross_entropy(z.reshape(-1,1024),b.reshape(-1),reduction='sum').item();n+=b.numel()
    srcres[lab][src]={'docs_sampled':min(64,len(groups[src])),'windows':len(seqs),'tokens':n,'loss':total/max(1,n)}
  del m;torch.cuda.empty_cache()
 out['source_validation_sample_loss']=srcres;save('source_validation_loss.json',out);print('PHASE3',srcres,flush=True)
 # P4 source-wise train sample repetition signatures; corpus remains unchanged
 reservoir=collections.defaultdict(list);seen=collections.Counter();rr=random.Random(420)
 for line in (V/'train_documents.jsonl').open(encoding='utf8'):
  d=json.loads(line);src=d.get('source','unknown');src='wiki_family' if src in ('wikipedia','wikibooks','wikinews') else src;text=d.get('text',d.get('cleaned_text',''));seen[src]+=1
  if len(reservoir[src])<100:reservoir[src].append(text)
  else:
   j=rr.randrange(seen[src])
   if j<100:reservoir[src][j]=text
 repeated={}
 for src,docs in reservoir.items():
  ng=collections.Counter();sent=collections.Counter();tc=collections.Counter();chars=0
  for text in docs:
   chars+=len(text)
   for ngram in (3,4):ng.update(text[i:i+ngram] for i in range(max(0,len(text)-ngram+1)))
   sent.update(x.strip() for x in re.split(r'[。！？!?\n]+',text) if len(x.strip())>=12)
   tc.update(tok.encode(text))
  repeated[src]={'sample_docs':len(docs),'total_docs':seen[src],'sample_chars':chars,'top_char_ngrams':[{'text':x,'count':n} for x,n in ng.most_common(20)],'repeated_sentences':[{'text':x[:200],'count':n} for x,n in sent.most_common(20) if n>1],'top_tokens':[{'id':int(i),'text':tok.decode([int(i)]),'count':n} for i,n in tc.most_common(30)]}
 out['train_repetition_sample']=repeated;save('data_repetition.json',out);print('PHASE4 complete',flush=True)
 # P5 fresh assistant-only SFT per Base; P6 fixed test only after best-val selection
 questions=json.loads((CHAT/'fixed_test_questions.json').read_text(encoding='utf8'));chat={}
 for lab,basepath in BASE.items():
  b=cp(basepath);cfg=ModelConfig(**b['model_config']);btok=BPETokenizer.from_dict(b['tokenizer_config'])
  tr=load_instruction_examples(CHAT/'train.jsonl',btok,128);va=load_instruction_examples(CHAT/'validation.jsonl',btok,128)
  m=TinyGPT(cfg);m.load_state_dict(b['model_state_dict']);del b;m.to('cuda',dtype=torch.float32).train()
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);br=torch.Generator().manual_seed(42);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  def evaluate(data):
   m.eval();tot=count=0
   with torch.inference_mode():
    for st in range(0,len(data),8):
     x,y,mask=collate_instruction_batch(data,list(range(st,min(st+8,len(data)))),'cuda');z=m(x).float();v=F.cross_entropy(z.reshape(-1,1024),y.reshape(-1),reduction='none').view_as(y).masked_select(mask);tot+=v.sum().item();count+=v.numel()
   m.train();val=tot/max(1,count)
   if not math.isfinite(val):raise FloatingPointError('nonfinite chat eval')
   return val
  hist=[{'update':0,'train_loss':evaluate(tr),'validation_loss':evaluate(va)}];best=hist[0]['validation_loss'];bs=0;cd=C/lab;cd.mkdir(exist_ok=False)
  def savebest(step):
   ob={'model_config':cfg.to_dict(),'model_state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'tokenizer_type':'bpe','tokenizer_config':btok.to_dict(),'step':step,'best_validation_loss':best,'base_checkpoint':str(basepath),'training_config':{'lr':1e-4,'batch_size':8,'seed':42,'precision':'FP32','gradient_clipping':1.0,'objective':'assistant-only','context':128,'fresh_from_base':True}}
   tmp=cd/'best.pt.tmp';torch.save(ob,tmp);tmp.replace(cd/'best.pt')
  savebest(0)
  for step in range(1,301):
   idx=torch.randint(len(tr),(8,),generator=br).tolist();x,y,mask=collate_instruction_batch(tr,idx,'cuda');opt.zero_grad(set_to_none=True);z=m(x);losses=F.cross_entropy(z.reshape(-1,1024),y.reshape(-1),reduction='none').view_as(y).masked_select(mask)
   if not losses.numel():raise ValueError('empty answer mask')
   loss=losses.mean()
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite chat loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
   if step%50==0:
    row={'update':step,'train_loss':evaluate(tr),'validation_loss':evaluate(va)};hist.append(row)
    if row['validation_loss']<best:best=row['validation_loss'];bs=step;savebest(step)
    print('CHAT',lab,step,row,'best',best,bs,flush=True)
  bestc=cp(cd/'best.pt');m.load_state_dict(bestc['model_state_dict']);m.eval();rows=[]
  for q in questions:
   prompt=format_prompt(q);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   with torch.inference_mode():full=generate(m,btok,prompt,max_new_tokens=120,temperature=.7,top_k=40)
   ans=(full[len(prompt):] if full.startswith(prompt) else full).split('<|user|>',1)[0].split('<|assistant|>',1)[0].strip();ids=btok.encode(ans);qids=set(btok.encode(q));aa=sum(c.isascii() and c.isalpha() for c in ans);ll=sum(c.isalpha() for c in ans)
   rows.append({'question':q,'answer':ans,'3gram_repetition':rep(ans,3),'4gram_repetition':rep(ans,4),'prompt_token_coverage':len(qids&set(ids))/max(1,len(qids)),'empty':not bool(ans),'replacement_char':'\ufffd' in full,'english_alpha_ratio':aa/max(1,ll)})
  chat[lab]={'history':hist,'best_update':bs,'best_validation_loss':best,'test_generations':rows,'mean_3gram_repetition':float(np.mean([x['3gram_repetition'] for x in rows])),'mean_4gram_repetition':float(np.mean([x['4gram_repetition'] for x in rows])),'empty_count':sum(x['empty'] for x in rows),'replacement_count':sum(x['replacement_char'] for x in rows),'english_outlier_count':sum(x['english_alpha_ratio']>.4 for x in rows),'best_checkpoint':str(cd/'best.pt')}
  out['chat_transfer']=chat;save('chat_transfer_results.json',out);del m,opt;torch.cuda.empty_cache();print('CHAT COMPLETE',lab,flush=True)
 out['elapsed_seconds']=time.perf_counter()-start;out['peak_vram_allocated_mib']=torch.cuda.max_memory_allocated()/1048576;out['peak_vram_reserved_mib']=torch.cuda.max_memory_reserved()/1048576
 save('final_report.json',out);print('ALL COMPLETE hours',out['elapsed_seconds']/3600,'peak MiB',out['peak_vram_allocated_mib'],'reserved',out['peak_vram_reserved_mib'],flush=True)
if __name__=='__main__':main()
