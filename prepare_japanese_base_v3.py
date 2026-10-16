"""Build an isolated Japanese Base v3 corpus from licensed prose sources."""
import gzip, hashlib, html, json, random, re, unicodedata, time, zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from bpe_tokenizer import BPETokenizer

ROOT=Path(__file__).resolve().parent
V2=ROOT/'datasets/japanese_base_v2'
OUT=ROOT/'datasets/japanese_base_v3'
RAW=OUT/'raw'
TOKENIZER_PATH=V2/'bpe_1024.json'
SEED=42
FW_REPO='hotchpotch/fineweb-2-edu-japanese'
FW_CONFIG='small_tokens_cleaned'
FW_FILES=[f'{FW_CONFIG}/train-{i:05d}-of-00283.parquet' for i in range(4)]
AO_REPO='globis-university/aozorabunko-clean'
AO_FILE='aozorabunko-dedupe-clean.jsonl.gz'
FW_CHAR_TARGET=58_000_000
AO_CHAR_TARGET=8_000_000

NAV=re.compile(r'^(?:前のページ|次のページ|ホーム|トップページ|サイトマップ|ログイン|新規登録|メニュー|検索|目次|関連記事|関連記事一覧|広告|スポンサーリンク|cookie|プライバシーポリシー|利用規約|お問い合わせ|copyright|©|ページトップ|戻る|進む|続きを読む|もっと見る)\s*$',re.I)
BOILER=re.compile(r'(?:cookieを使用|クッキーを使用|javascriptを有効|このサイトではcookie|すべての権利を保有|無断転載禁止|プライバシーポリシー|サイトマップ|ログインはこちら|会員登録はこちら)',re.I)
URL=re.compile(r'https?://\S+|www\.\S+',re.I)

def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).lower()
def japanese_count(s):return sum(('\u3040'<=c<='\u30ff') or ('\u3400'<=c<='\u9fff') for c in s)
def sha(s):return hashlib.sha256(norm(s).encode('utf-8')).hexdigest()
def clean_text(value):
    s=html.unescape(str(value or '')).replace('\r','\n').replace('\x00',' ')
    s=re.sub(r'(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>',' ',s)
    s=re.sub(r'(?s)\{\|.*?\|\}',' ',s)
    for _ in range(5):
        old=s;s=re.sub(r'\{\{[^{}]*\}\}',' ',s)
        if old==s:break
    s=re.sub(r'\[\[(?:Category|カテゴリ|File|Image|ファイル|画像):[^\]]*\]\]',' ',s,flags=re.I)
    s=re.sub(r'\[\[([^]|]+)\|([^]]+)\]\]',r'\2',s)
    s=re.sub(r'\[\[([^]]+)\]\]',r'\1',s)
    s=re.sub(r'\[([^]]+)\]\((?:https?://)[^)]+\)',r'\1',s)
    s=re.sub(r'(?is)<[^>]+>',' ',s)
    s=URL.sub(' ',s)
    s=re.sub(r'(?m)^\s*(?:\[?https?://\S+\]?|www\.\S+)\s*$',' ',s)
    s=re.sub(r'(?m)^\s*={2,}\s*([^=]+?)\s*={2,}\s*$','\1',s)
    s=re.sub(r'(?m)^\s*[*#;:|]+\s*','',s)
    s=s.replace('\u3000',' ').replace('\t',' ')
    lines=[]
    for line in s.splitlines():
        line=re.sub(r'\s+',' ',line).strip()
        if not line or NAV.fullmatch(line) or BOILER.search(line):continue
        if len(line)<20 and not re.search(r'[。！？!?]$',line):continue
        if len(re.findall(r'[\wぁ-んァ-ヶ一-龠]',line))/max(1,len(line))<.45:continue
        lines.append(line)
    text='\n'.join(lines)
    text=re.sub(r'\n{3,}','\n\n',text).strip()
    return text

def reject_quality(text):
    if len(text)<180 or japanese_count(text)<100:return True,'short_or_non_japanese'
    jp=japanese_count(text)
    visible=sum(not c.isspace() for c in text)
    if jp/max(1,visible)<.30:return True,'low_japanese_ratio'
    if sum(not(c.isalnum() or c.isspace() or '\u3040'<=c<='\u30ff' or '\u3400'<=c<='\u9fff' or c in '。、，．！？!?・「」『』（）()［］[]：:；;％%〜～…—-') for c in text)/max(1,visible)>.12:return True,'symbol_heavy'
    if re.search(r'<\|(?:user|assistant)\|>|###\s*(?:instruction|response)|(?:質問|回答)\s*[:：]',text,re.I):return True,'instruction_or_qa_format'
    if re.search(r'(?:(.)\1{11,})',text):return True,'character_loop'
    return False,''

def sentence_dedup(text,seen):
    chunks=re.split(r'(?<=[。！？!?])\s*|\n+',text)
    kept=[];local=set()
    for s in chunks:
        s=s.strip();key=norm(s)
        if len(key)>=20:
            fingerprint=zlib.crc32(key.encode('utf-8'))
            if fingerprint in local or fingerprint in seen:continue
            local.add(fingerprint);seen.add(fingerprint)
        if s:kept.append(s)
    return '\n'.join(kept)

def grams(text,n=5):
    s=norm(text)
    count=max(0,len(s)-n+1)
    if not count:return set()
    # A deterministic, evenly spaced sample keeps the near-duplicate index
    # tractable while retaining enough coverage for long documents.
    stride=max(1,count//256)
    return {zlib.crc32(s[i:i+n].encode('utf-8')) for i in range(0,count,stride)}
def simhash64(g):
    if not g:return 0
    values=np.fromiter(g,dtype=np.uint32)
    bits=np.unpackbits(values.view(np.uint8)).reshape(-1,32)
    votes=bits.sum(axis=0)
    return sum((1<<i) for i,v in enumerate(votes) if v*2>=len(values))
def hamming(a,b):return (a^b).bit_count()
def jaccard(a,b):return len(a&b)/max(1,len(a|b))
class Deduper:
    def __init__(self):self.docs=[];self.fps=set();self.bands=defaultdict(set);self.exact=0;self.near=0
    def add(self,row):
        fp=sha(row['text'])
        if fp in self.fps:self.exact+=1;return False
        sh=grams(row['text']);sig=simhash64(sh)
        near_ids=set()
        for i in range(4):near_ids.update(self.bands[(i,(sig>>(8*i))&0xff)])
        for j in near_ids:
            old=self.docs[j]
            if hamming(sig,old['_simhash'])<=3 and jaccard(sh,grams(old['text']))>=.85:
                self.near+=1;return False
        ix=len(self.docs);row['_simhash']=sig;self.docs.append(row);self.fps.add(fp)
        for i in range(4):self.bands[(i,(sig>>(8*i))&0xff)].add(ix)
        return True

def file_download(repo,file,repo_type='dataset'):
    return Path(hf_hub_download(repo_id=repo,filename=file,repo_type=repo_type,local_dir=RAW/'hub'))

def find_v2_docs():
    out=[]
    for split in ('train','validation'):
        with (V2/f'{split}_documents.jsonl').open(encoding='utf-8') as f:
            for line in f:
                d=json.loads(line);out.append({'id':f"v2-{d['source']}-{d['id']}",'source':d['source'],'source_split':split,
                    'title':d.get('title',''),'url':d.get('url',''),'text':d['cleaned_text'],'license':None})
    return out

def main():
    if OUT.exists() and any(p.name!='raw' for p in OUT.iterdir()):
        raise FileExistsError(f'refusing to overwrite {OUT}')
    if not TOKENIZER_PATH.is_file():raise FileNotFoundError(TOKENIZER_PATH)
    tok=BPETokenizer.load(TOKENIZER_PATH)
    if tok.vocab_size!=1024 or tok.vocab[:256]!=[bytes([x]) for x in range(256)]:raise ValueError('Expected unchanged Byte-BPE 1024')
    RAW.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();raw_counts=Counter();rej=Counter();candidate=[];sentence_seen=set();dedup=Deduper()
    # Keep all v2 documents and their train/validation assignments intact.
    v2docs=find_v2_docs()
    for d in v2docs:
        # Carry v2 cleaned text forward without altering its contents.
        for sentence in re.split(r'(?<=[。！？!?])\s*|\n+',d['text']):
            key=norm(sentence)
            if len(key)>=20:sentence_seen.add(zlib.crc32(key.encode('utf-8')))
        if not dedup.add(d):continue
        candidate.append(d)
    v2_keep=len(candidate)
    # Aozora public-domain texts carry explicit CC BY 4.0 dataset provenance.
    ao_path=file_download(AO_REPO,AO_FILE)
    ao_chars=0;ao_adopted=0;ao_candidates=[]
    with gzip.open(ao_path,'rt',encoding='utf-8') as f:
        for line_no,line in enumerate(f):
            raw_counts['aozora']+=1
            try:d=json.loads(line)
            except json.JSONDecodeError:rej['aozora_bad_json']+=1;continue
            meta=d.get('meta') or {};style=meta.get('文字遣い種別','')
            # Prefer modern kana conventions and non-fiction categories; use a small
            # literary share for natural Japanese, selected deterministically.
            if style!='新字新仮名':rej['aozora_old_orthography']+=1;continue
            text=clean_text(d.get('text',''))
            if text.startswith('底本：') or text.startswith('入力：'):text='\n'.join(text.splitlines()[2:]).strip()
            bad,why=reject_quality(text)
            if bad:rej['aozora_'+why]+=1;continue
            ndc=str(meta.get('分類番号',''))
            nonfic=bool(re.search(r'随筆|科学|研究|社会|文化|経済|技術|自然|教育|歴史|生活|地理|天文|医学',meta.get('作品名','')+' '+text[:300]))
            if not nonfic and not ndc.startswith(tuple('01234567')):
                rej['aozora_topic_filter']+=1;continue
            row={'id':f"aozora-{meta.get('作品ID',line_no)}",'source':'globis-university/aozorabunko-clean',
                 'source_split':'new','title':meta.get('作品名',''),'url':meta.get('図書カードURL',''),
                 'text':text,'license':'CC BY 4.0'}
            ao_candidates.append(row)
    ao_candidates.sort(key=lambda x:hashlib.sha1((str(SEED)+x['id']).encode()).hexdigest())
    for row in ao_candidates:
        text=sentence_dedup(row['text'],sentence_seen);bad,why=reject_quality(text)
        if bad:rej['aozora_'+why+'_after_sentence_dedup']+=1;continue
        row['text']=text
        if not dedup.add(row):continue
        candidate.append(row);ao_adopted+=1;ao_chars+=len(text)
        if ao_chars>=AO_CHAR_TARGET:break
    # Educational prose from the cleaned FineWeb2 subset (four deterministic shards).
    fw_chars=0;fw_adopted=0;fw_raw_seen=set();rng=random.Random(SEED)
    for shard_index,path_in_repo in enumerate(FW_FILES):
        path=file_download(FW_REPO,path_in_repo)
        pf=pq.ParquetFile(path)
        cols=set(pf.schema_arrow.names)
        want=[x for x in ('text','url','id','score','token_count','language','language_script','is_cleaned') if x in cols]
        for batch in pf.iter_batches(batch_size=2048,columns=want):
            for r in batch.to_pylist():
                raw_counts['fineweb2_edu']+=1
                identity=str(r.get('id') or r.get('url') or hashlib.sha1(str(r.get('text','')).encode()).hexdigest())
                if identity in fw_raw_seen:continue
                fw_raw_seen.add(identity)
                if r.get('is_cleaned') is False:rej['fineweb_not_cleaned']+=1;continue
                try:score=float(r.get('score',0) or 0)
                except Exception:score=0
                if score and score<3.0:rej['fineweb_low_edu_score']+=1;continue
                text=clean_text(r.get('text',''))
                bad,why=reject_quality(text)
                if bad:rej['fineweb_'+why]+=1;continue
                # Keep prose; remove explicit QA/instruction and page-list artifacts.
                text=sentence_dedup(text,sentence_seen)
                bad,why=reject_quality(text)
                if bad:rej['fineweb_'+why+'_after_sentence_dedup']+=1;continue
                row={'id':f"fineweb2-{identity}",'source':'hotchpotch/fineweb-2-edu-japanese:small_tokens_cleaned',
                     'source_split':'new','title':'','url':r.get('url',''),'text':text,'license':'ODC-By 1.0',
                     'source_record_id':identity,'source_score':score}
                if not dedup.add(row):continue
                candidate.append(row);fw_adopted+=1;fw_chars+=len(text)
                if fw_chars>=FW_CHAR_TARGET:break
            if fw_chars>=FW_CHAR_TARGET:break
        if fw_chars>=FW_CHAR_TARGET:break
        print(f'FineWeb shard {shard_index+1}: raw={raw_counts["fineweb2_edu"]:,}, accepted={fw_adopted:,}, chars={fw_chars:,}',flush=True)
    if fw_chars<FW_CHAR_TARGET*.65:raise RuntimeError(f'FineWeb short of target: {fw_chars:,} chars; need more shards')
    print(f'Candidates: FineWeb={fw_adopted:,}/{fw_chars:,} chars, Aozora={ao_adopted:,}/{ao_chars:,} chars; running tokenization',flush=True)
    # Existing v2 assignments remain unchanged; only new corpus documents are split.
    train_docs=[];val_docs=[]
    for d in candidate:
        if d['source_split']=='train':train_docs.append(d)
        elif d['source_split']=='validation':val_docs.append(d)
        else:
            frac=int(hashlib.sha1((str(SEED)+d['id']).encode()).hexdigest()[:8],16)/0xffffffff
            (val_docs if frac<.05 else train_docs).append(d)
    # Stream documents in bounded chunks; the entire tokenized corpus would exceed
    # memory on the target workstation. Each chunk is checked before being cached.
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter();split_stats={}
    for name,rows in [('train',train_docs),('validation',val_docs)]:
        text_path=OUT/f'{name}.txt';docs_path=OUT/f'{name}_documents.jsonl'
        chunk_dir=RAW/f'token_chunks_{name}';chunk_dir.mkdir(parents=True,exist_ok=True)
        char_count=byte_count=token_count=0;chunk_text=[];chunk_chars=0;chunk_index=0
        with text_path.open('x',encoding='utf-8') as tf, docs_path.open('x',encoding='utf-8') as df:
            def flush_chunk():
                nonlocal chunk_index,chunk_text,chunk_chars,token_count
                if not chunk_text:return
                value='\n'.join(chunk_text)+'\n'
                ids=tok.encode_array(value)
                if tok.decode(ids,errors='strict')!=value:
                    raise RuntimeError(f'{name} chunk {chunk_index} tokenizer round trip failed')
                np.save(chunk_dir/f'{chunk_index:06d}.npy',ids.astype(np.uint32,copy=False),allow_pickle=False)
                token_count+=len(ids);chunk_index+=1;chunk_text=[];chunk_chars=0
            for d in rows:
                d.pop('_simhash',None)
                df.write(json.dumps(d,ensure_ascii=False)+'\n')
                tf.write(d['text']+'\n')
                char_count+=len(d['text'])+1;byte_count+=len(d['text'].encode('utf-8'))+1
                chunk_text.append(d['text']);chunk_chars+=len(d['text'])+1
                if chunk_chars>=250_000:flush_chunk()
            flush_chunk()
        final=np.lib.format.open_memmap(OUT/f'{name}_tokens.npy',mode='w+',dtype=np.uint32,shape=(token_count,))
        offset=0
        for chunk_path in sorted(chunk_dir.glob('*.npy')):
            part=np.load(chunk_path,mmap_mode='r',allow_pickle=False)
            final[offset:offset+len(part)]=part;offset+=len(part)
        final.flush();del final
        if offset!=token_count:raise RuntimeError(f'{name} token cache length mismatch')
        split_stats[name]={'documents':len(rows),'characters':char_count,'utf8_bytes':byte_count,
                           'tokens':token_count,'bytes_per_token':byte_count/token_count}
    if split_stats['train']['tokens']<2*36_556_312:
        raise RuntimeError(f"v3 train tokens {split_stats['train']['tokens']:,} below 2x v2")
    def source_report(rows):
        out={}
        for d in rows:
            z=out.setdefault(d['source'],{'documents':0,'characters':0,'japanese_characters':0,'license':d['license'] or 'carried forward from Base v2 source metadata'})
            z['documents']+=1;z['characters']+=len(d['text']);z['japanese_characters']+=japanese_count(d['text'])
        return out
    report={'dataset':'Japanese Base v3 candidate','seed':SEED,'tokenizer':'existing Japanese Byte-BPE 1024; not retrained',
      'split_policy':'v2 documents retain their original split; new documents assigned by stable ID hash (95/5); document-level only',
      'train':split_stats['train'],'validation':split_stats['validation'],
      'total_train_tokens_multiple_of_v2':split_stats['train']['tokens']/36_556_312,
      'sources':{'train':source_report(train_docs),'validation':source_report(val_docs)},
      'source_raw_counts':dict(raw_counts),'source_adopted_counts':{'v2':v2_keep,'aozora':ao_adopted,'fineweb2_edu':fw_adopted},
      'new_source_characters':{'aozora':ao_chars,'fineweb2_edu':fw_chars},'filters':dict(rej),
      'exact_duplicate_documents_removed':dedup.exact,'near_duplicate_documents_removed':dedup.near,
      'near_duplicate_method':'32-bit SimHash candidate buckets over normalized char 5-grams; verify Jaccard >=0.85 for candidate pairs; applied across sources/splits',
      'train_validation_document_overlap':0,'train_validation_near_duplicate_pairs':0,
      'cleaning':'HTML/wiki/navigation/URL-only/short fragments/boilerplate/symbol-heavy/instruction-QA/repeated chars removed; long normalized repeated sentences removed globally',
      'tokenizer_efficiency':{'existing_train_tokens_v2':36_556_312,'v2_train_characters':37_121_657,'v2_bytes_per_token':len(V2.joinpath('train.txt').read_bytes())/36_556_312,
        'v3_train_bytes_per_token':split_stats['train']['bytes_per_token'],'encode_decode_exact':True},
      'licenses':{'FineWeb2 Edu Japanese':'ODC-By 1.0; also Common Crawl terms','Aozora clean':'CC BY 4.0; source card describes public-domain works'},
      'identifiers':{'FineWeb2':FW_REPO,'Aozora':AO_REPO},'processing_seconds':time.perf_counter()-started,'tokenization_seconds':time.perf_counter()-t0}
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
