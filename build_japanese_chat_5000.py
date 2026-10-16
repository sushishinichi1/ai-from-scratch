"""Expand the preserved 1000-example split with filtered Self-Instruct examples."""
import json, re, random, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import build_japanese_chat_1000 as curate
from bpe_tokenizer import BPETokenizer
from chat_data import format_prompt

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'datasets/japanese_base_v2/raw/instruction_sources/GPT-4-Self-Instruct-Japanese.json'
OLD=ROOT/'datasets/japanese_base_v2/chat_1000'
OUT=ROOT/'datasets/japanese_base_v2/chat_5000'
TOK=ROOT/'datasets/japanese_base_v2/bpe_1024.json'
SEED=42

def norm(s): return re.sub(r'[^\wぁ-んァ-ヶ一-龠]', '', unicodedata.normalize('NFKC',s).lower())
def grams(s):
    n=norm(s); return {n[i:i+3] for i in range(max(0,len(n)-2))}
def jac(a,b): return len(a&b)/max(1,len(a|b))
def domain(q,a): return curate.domain_for(q+' '+a)
def task(q): return curate.task_for(q)
def semantic_overlap(q,a):
    qg,ag=grams(q),grams(a)
    if not qg or not ag:return 0.
    return len(qg&ag)/len(qg)
def fits(q,a,tok):
    p=format_prompt(q); serialized=p+a+'\n'
    if not q.strip() or not a.strip() or '\ufffd' in q+a:return False
    if len(a)>180 or re.search(r'https?://|www\.|<\s*(?:div|script|table|code)|```|\{\{|\}\}',q+a,re.I):return False
    if re.fullmatch(r'[\W\d_]+',q,re.UNICODE) or re.fullmatch(r'[\W\d_]+',a,re.UNICODE):return False
    if semantic_overlap(q,a)<0.055:return False
    ids=tok.encode(p)+tok.encode(a)
    return len(ids)<=128 and len(tok.encode(serialized))<=128 and tok.decode(tok.encode(serialized))==serialized
def main():
    if OUT.exists(): raise FileExistsError(f'refusing to overwrite {OUT}')
    tok=BPETokenizer.load(TOK)
    parts={name:[json.loads(x) for x in (OLD/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()] for name in ('train','validation','test')}
    if tuple(map(len,(parts['train'],parts['validation'],parts['test'])))!=(800,100,100):raise ValueError('old split size mismatch')
    oldrows=sum(parts.values(),[]); fixed=json.loads((OLD/'fixed_test_questions.json').read_text(encoding='utf-8'))
    # Preserve every prior record and its original split unchanged.
    old_questions=[x['question'] for x in oldrows]; exact={norm(q) for q in old_questions}
    old_grams=[grams(q) for q in old_questions]
    rng=random.Random(SEED); raw=json.loads(SRC.read_text(encoding='utf-8')); rng.shuffle(raw)
    candidates=[]; rejected=Counter(); seen_ids=set(); indexes=defaultdict(set)
    for r in old_grams:
        pass
    for ix,item in enumerate(raw):
        instr=str(item.get('instruction','')).strip(); context=str(item.get('input','')).strip()
        q=instr+('\n'+context if context else '')
        if not q: rejected['empty_question']+=1; continue
        # Preserve the original task wording where it fits; otherwise try a
        # concise topic-anchored form to recover context-bound examples.
        options=[q,curate.compact_question(instr,context,str(item.get('output','')),ix)]
        chosen=None
        for option in dict.fromkeys(options):
            for a in curate.answer_candidates(item.get('output',''),option):
                if not fits(option,a,tok):continue
                key=norm(option)
                if key in exact: rejected['duplicate_existing_question']+=1; continue
                qg=grams(option)
                if any(jac(qg,g)>=0.78 for g in old_grams): rejected['near_duplicate_existing']+=1; continue
                # Short/boilerplate responses and source-answer mismatches fail the overlap test above.
                chosen=(option,a,qg); break
            if chosen:break
        if not chosen: rejected['quality_context_or_duplicate_filter']+=1; continue
        option,a,qg=chosen
        # Candidate deduplication via 3-gram inverted index avoids quadratic full scans.
        nearby=set()
        for g in qg: nearby.update(indexes.get(g,()))
        duplicate=False
        for ci in nearby:
            if jac(qg,candidates[ci]['_grams'])>=0.78: duplicate=True;break
        if duplicate: rejected['near_duplicate_source']+=1;continue
        row={'id':f'ja5k-gpt4-{ix:05d}','category':domain(option,a),'task_type':task(option),
             'source':'CausalLM/GPT-4-Self-Instruct-Japanese@9178a2d','source_record_id':ix,
             'question':option,'answer':a,'text':format_prompt(option)+a+'\n'}
        ci=len(candidates);candidates.append({**row,'_grams':qg});
        for g in qg:indexes[g].add(ci)
        exact.add(norm(option))
        if len(candidates)>=4000:break
    if len(candidates)<4000:raise RuntimeError(f'Only {len(candidates)} valid examples; need 4000, rejected={dict(rejected)}')
    candidates=candidates[:4000]
    # Stable target counts: 3200 new train, 400 validation, 400 test.
    rng.shuffle(candidates)
    extra={'train':candidates[:3200],'validation':candidates[3200:3600],'test':candidates[3600:]}
    for split in extra:
        for row in extra[split]:row.pop('_grams',None)
        parts[split].extend(extra[split])
    # Check all question near-duplicates across each split, including protected prior rows.
    sets={k:[grams(x['question']) for x in v] for k,v in parts.items()}
    cross=[]
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        idx=defaultdict(list)
        for j,g in enumerate(sets[b]):
            for item in g:idx[item].append(j)
        for i,g in enumerate(sets[a]):
            ids=set()
            for item in g:ids.update(idx.get(item,()))
            for j in ids:
                if jac(g,sets[b][j])>=0.78:
                    cross.append([a,i,b,j]);
                    if len(cross)>0:break
            if cross:break
        if cross:break
    if cross:raise RuntimeError(f'cross split near duplicate: {cross[0]}')
    allrows=sum(parts.values(),[]); pairkeys=[(norm(x['question']),norm(x['answer'])) for x in allrows]
    if len(pairkeys)!=len(set(pairkeys)):raise RuntimeError('exact duplicate remained')
    if any(norm(q) in {norm(f) for f in fixed} for q in [x['question'] for x in parts['train']+parts['validation']]):raise RuntimeError('fixed test leaked')
    if (len(parts['train']),len(parts['validation']),len(parts['test']))!=(4000,500,500):raise RuntimeError('split count mismatch')
    # Verify every row is valid and context bounded with the exact unchanged tokenizer.
    maxctx=0
    for row in allrows:
        p=format_prompt(row['question']); assert row['text']==p+row['answer']+'\n'
        assert tok.decode(tok.encode(row['text']))==row['text']
        n=max(len(tok.encode(p))+len(tok.encode(row['answer'])),len(tok.encode(row['text'])))
        if n>128:raise RuntimeError('context violation')
        maxctx=max(maxctx,n)
        if not row['question'] or not row['answer']:raise RuntimeError('empty field')
    OUT.mkdir(parents=True)
    for name,rows in parts.items():
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    (OUT/'fixed_test_questions.json').write_text(json.dumps(fixed,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'total_records':len(allrows),'train':4000,'validation':500,'test':500,'retained_existing_1000':True,'retained_existing_108':108,
            'new_source_records':4000,'source':'CausalLM/GPT-4-Self-Instruct-Japanese','source_revision':'9178a2d','license':'CC BY 4.0',
            'fixed_test_count':len(fixed),'fixed_test_test_only':True,'tokenizer':'Japanese Base v2 Byte-BPE 1024 (unchanged)',
            'exact_duplicate_questions_or_pairs':0,'cross_split_near_duplicate_count':0,'near_duplicate_method':'normalized character 3-gram Jaccard >=0.78',
            'context_limit':128,'maximum_context_tokens':maxctx,'all_utf8':True,'empty_questions_or_answers':0,'filters':dict(rejected),
            'source_provenance_fields':['source','source_record_id'],'category_counts':dict(Counter(x['category'] for x in allrows)),
            'task_counts':dict(Counter(x['task_type'] for x in allrows))}
    (OUT/'dataset_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
