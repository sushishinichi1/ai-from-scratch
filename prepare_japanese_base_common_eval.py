"""Create a frozen, held-out Japanese common evaluation set for Base v2/v3."""
import gzip, hashlib, html, json, re, unicodedata, zlib
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from bpe_tokenizer import BPETokenizer
from prepare_japanese_base_v3 import clean_text, reject_quality

ROOT=Path(__file__).resolve().parent
V2=ROOT/'datasets/japanese_base_v2'
V3=ROOT/'datasets/japanese_base_v3'
OUT=ROOT/'datasets/japanese_base_common_eval'
RAW=V3/'raw/hub'
SEED=42
FW='hotchpotch/fineweb-2-edu-japanese'
AO='globis-university/aozorabunko-clean'
TOKENIZER=V2/'bpe_1024.json'

TERMS={
 'science':('科学','研究','実験','生物','物理','化学','医学','宇宙','地球','植物','動物','細胞','エネルギー'),
 'technology':('コンピュータ','技術','ソフトウェア','インターネット','プログラム','機械','電気','通信','デジタル','情報処理'),
 'society':('社会','経済','教育','政治','法律','地域','企業','仕事','制度','人口','福祉'),
 'culture':('文化','音楽','芸術','文学','映画','伝統','歴史','祭り','言語','スポーツ'),
 'nature':('自然','環境','森林','海','川','山','気候','気象','農業','食物','生態'),
}
QUOTAS={'science':100,'technology':100,'society':90,'culture':90,'nature':90,'general':130}

def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).lower()
def grams(s):
 s=norm(s);n=max(0,len(s)-4)
 if not n:return set()
 stride=max(1,n//64)
 return {zlib.crc32(s[i:i+5].encode()) for i in range(0,n,stride)}
def simhash(g):
 if not g:return 0
 a=np.fromiter(g,dtype=np.uint32);bits=np.unpackbits(a.view(np.uint8)).reshape(-1,32).sum(0)
 return sum((1<<i) for i,x in enumerate(bits) if x*2>=len(a))
def category(s):
 for name,words in TERMS.items():
  if any(w in s for w in words):return name
 return 'general'
def hash_id(s):return hashlib.sha1((str(SEED)+s).encode()).hexdigest()

class NearIndex:
 def __init__(self):self.exact=set();self.bands=defaultdict(list);self.sig=[];self.features=[];self.near=0;self.exact_rejected=0
 def add(self,row,training=False):
  text=row['text'];key=hashlib.sha256(norm(text).encode()).digest()
  if key in self.exact:self.exact_rejected+=1;return False
  g=grams(text);sig=simhash(g);ids=set()
  for b in range(4):ids.update(self.bands[(b,(sig>>(8*b))&255)])
  for j in ids:
   if (sig^self.sig[j]).bit_count()<=3:
    a=self.features[j];inter=len(g.intersection(a))
    if inter/max(1,len(g)+len(a)-inter)>=.80:
     self.near+=1;return False
  i=len(self.sig);self.exact.add(key);self.sig.append(sig);self.features.append(g)
  for b in range(4):self.bands[(b,(sig>>(8*b))&255)].append(i)
  return True

def main():
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'refusing to overwrite {OUT}')
 tok=BPETokenizer.load(TOKENIZER)
 if tok.vocab_size!=1024:raise ValueError('expected Byte-BPE 1024')
 index=NearIndex();v3_ids=set();train_count=0
 # v3 train already carries every v2 train document; verify this before relying on it.
 with (V3/'train_documents.jsonl').open(encoding='utf8') as f:
  for line in f:
   d=json.loads(line);v3_ids.add(d['id']);train_count+=1
   row={'text':d.get('text',d.get('cleaned_text',''))}
   index.add(row,training=True)
 v2_train_ids=set()
 with (V2/'train_documents.jsonl').open(encoding='utf8') as f:
  for line in f:v2_train_ids.add('v2-'+(lambda d:f"{d['source']}-{d['id']}")(json.loads(line)))
 missing=v2_train_ids-v3_ids
 if missing:raise RuntimeError(f'{len(missing)} v2 train documents missing from v3 train')
 print(f'Indexed v3 train documents={train_count:,}; all v2 train docs included',flush=True)
 # Aozora candidates, limited to public-domain works in modern orthography.
 ao_path=RAW/'aozorabunko-dedupe-clean.jsonl.gz';ao_pool=[];ao_raw=0
 trained_ao={d.split('-',1)[1] for d in v3_ids if d.startswith('aozora-')}
 with gzip.open(ao_path,'rt',encoding='utf8') as f:
  for ln,line in enumerate(f):
   ao_raw+=1
   try:d=json.loads(line)
   except json.JSONDecodeError:continue
   m=d.get('meta') or {};wid=str(m.get('作品ID',''))
   if m.get('文字遣い種別')!='新字新仮名' or m.get('作品著作権フラグ')!='なし' or wid in trained_ao:continue
   text=clean_text(d.get('text',''));bad,_=reject_quality(text)
   if bad or len(text)>20000:continue
   ao_pool.append({'id':'aozora-'+wid,'source':'Aozora Bunko','url':m.get('図書カードURL',''),
     'title':m.get('作品名',''),'category':'culture','text':text,'license':'CC BY 4.0'})
 # Stable selection, then fill from additional FineWeb training shards not used for v3.
 ao_pool.sort(key=lambda r:hash_id(r['id']))
 fw_pools={k:[] for k in (*QUOTAS,'other')};fw_raw=0
 for shard in (4,5):
  file=f'small_tokens_cleaned/train-{shard:05d}-of-00283.parquet'
  path=Path(hf_hub_download(repo_id=FW,filename=file,repo_type='dataset',local_dir=RAW.parent))
  import pyarrow.parquet as pq
  pf=pq.ParquetFile(path);cols=set(pf.schema_arrow.names)
  want=[c for c in ('text','url','id','score','is_cleaned') if c in cols]
  for batch in pf.iter_batches(batch_size=2048,columns=want):
   for r in batch.to_pylist():
    fw_raw+=1
    if r.get('is_cleaned') is False:continue
    try:score=float(r.get('score',0) or 0)
    except Exception:score=0
    if score and score<3:continue
    text=clean_text(r.get('text',''));bad,_=reject_quality(text)
    if bad or len(text)>12000:continue
    cat=category(text);identity=str(r.get('id') or r.get('url') or hashlib.sha1(text.encode()).hexdigest())
    fw_pools[cat if cat in QUOTAS else 'other'].append({'id':'fineweb-'+identity,
      'source':'FineWeb2 Edu Japanese small_tokens_cleaned held-out shards','url':r.get('url',''),
      'title':'','category':cat,'text':text,'license':'ODC-By 1.0'})
 # Top up category pools if any quota is short, then select 700 FineWeb docs.
 fw_selected=[];used=set()
 for cat,n in QUOTAS.items():
  pool=sorted(fw_pools[cat],key=lambda r:hash_id(r['id']))
  chosen=pool[:n];fw_selected.extend(chosen);used.update(x['id'] for x in chosen)
 leftovers=[]
 for pool in fw_pools.values():leftovers.extend(r for r in pool if r['id'] not in used)
 leftovers.sort(key=lambda r:hash_id(r['id']))
 fw_selected.extend(leftovers[:max(0,700-len(fw_selected))])
 ao_selected=ao_pool[:200]
 candidates=fw_selected+ao_selected
 # Exclude exact/long near matches against all existing v3 train documents and
 # against already accepted eval documents. Search extra candidates if necessary.
 accepted=[]
 for row in candidates:
  if index.add(row):accepted.append(row)
 if len(accepted)<500:
  backup=leftovers[700:]+ao_pool[200:]
  backup.sort(key=lambda r:hash_id(r['id']))
  for row in backup:
   if index.add(row):accepted.append(row)
   if len(accepted)>=500:break
 if not 500<=len(accepted)<=1000:raise RuntimeError(f'common eval size outside 500-1000: {len(accepted)}')
 OUT.mkdir(parents=True,exist_ok=True);all_ids=[];offsets=[0];per_source=Counter();per_cat=Counter();records=[]
 for i,row in enumerate(accepted):
  payload=row['text']+'\n';ids=tok.encode_array(payload)
  if tok.decode(ids,errors='strict')!=payload:raise RuntimeError(f'round trip failed for {row["id"]}')
  all_ids.extend(map(int,ids));offsets.append(len(all_ids));per_source[row['source']]+=1;per_cat[row['category']]+=1
  records.append({**row,'token_count':len(ids),'token_ids':ids.tolist()})
 with (OUT/'documents.jsonl').open('x',encoding='utf8') as f:
  for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 np.save(OUT/'tokens.npy',np.asarray(all_ids,dtype=np.uint32),allow_pickle=False)
 np.save(OUT/'offsets.npy',np.asarray(offsets,dtype=np.uint64),allow_pickle=False)
 manifest={'purpose':'fixed evaluation only; never train or select checkpoints with it',
  'seed':SEED,'documents':len(records),'tokens':len(all_ids),'tokenizer':str(TOKENIZER),
  'sources':dict(per_source),'categories':dict(per_cat),'raw_records_processed':{'aozora':ao_raw,'fineweb_heldout_shards':fw_raw},
  'fineweb_shards':['small_tokens_cleaned/train-00004-of-00283.parquet','small_tokens_cleaned/train-00005-of-00283.parquet'],
  'licenses':{'Aozora Bunko':'CC BY 4.0; dataset filters to public-domain works','FineWeb2 Edu Japanese':'ODC-By 1.0; Common Crawl terms also apply'},
  'split_exclusion':'v3 train documents include all v2 train documents; candidate source records were also filtered against v3 train',
  'v2_train_documents_missing_from_v3':len(missing),'exact_or_long_near_duplicate_candidate_rejections':index.exact_rejected+index.near,
  'exact_document_duplicates_rejected':index.exact_rejected,'near_document_duplicates_rejected':index.near,
  'near_method':'normalized text exact SHA-256 plus 32-bit SimHash over sampled normalized char 5-grams; Jaccard >=0.80 for SimHash distance <=3',
  'generation_or_training_use':'for evaluation only; excluded from optimizer and checkpoint selection'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
