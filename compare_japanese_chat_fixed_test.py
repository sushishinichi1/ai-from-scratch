"""Regenerate the final fixed-test comparison using the continued 1000-data best."""
import json
from pathlib import Path
import torch
from bpe_tokenizer import BPETokenizer
from model import ModelConfig,TinyGPT
from train_japanese_chat_5000 import generate_row,human_flags

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'checkpoints/japanese_base_v2/wide30m_lr0003_bpe1024/best.pt'
CHAT1K=ROOT/'checkpoints/japanese_base_v2/chat30m_1000_continue500_lr0001/best.pt'
CHAT5K=ROOT/'checkpoints/japanese_base_v2/chat30m_5000_lr0001/best.pt'
TOK=BPETokenizer.load(ROOT/'datasets/japanese_base_v2/bpe_1024.json')
FIXED=json.loads((ROOT/'datasets/japanese_base_v2/chat_5000/fixed_test_questions.json').read_text(encoding='utf-8'))
LOG=ROOT/'logs/japanese_base_v2/chat30m_long_run'
def load(path):
 c=torch.load(path,map_location='cpu',weights_only=False);cfg=ModelConfig(**c['model_config'])
 if c['tokenizer_config']!=TOK.to_dict():raise ValueError('tokenizer mismatch')
 m=TinyGPT(cfg);m.load_state_dict(c['model_state_dict']);m.to('cuda',dtype=torch.float32).eval();return m
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 models={'base':(BASE,load(BASE)),'1000_best':(CHAT1K,load(CHAT1K)),'5000_best':(CHAT5K,load(CHAT5K))}
 result={'settings':{'temperature':.7,'top_k':40,'max_new_tokens':120,'seed':42,'utf8_safe':True},'questions':FIXED,'models':{}}
 for name,(path,model) in models.items():
  rows=human_flags([generate_row(model,TOK,q) for q in FIXED])
  result['models'][name]={'checkpoint':str(path),'generations':rows,
   'auto_natural_answer_count':sum(x['auto_natural_answer_proxy'] for x in rows),
   'mean_3gram_repetition_rate':sum(x['3gram_repetition_rate'] for x in rows)/len(rows),
   'mean_4gram_repetition_rate':sum(x['4gram_repetition_rate'] for x in rows)/len(rows)}
  print(name,'done',flush=True)
 (LOG/'fixed_test20_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 p=LOG/'phase4_phase5_5000.json';summary=json.loads(p.read_text(encoding='utf-8'))
 summary['comparison_json']=str(LOG/'fixed_test20_comparison.json')
 summary['comparison_summary']={k:{'auto_natural_answer_count':v['auto_natural_answer_count'],'mean_3gram_repetition_rate':v['mean_3gram_repetition_rate'],'mean_4gram_repetition_rate':v['mean_4gram_repetition_rate']} for k,v in result['models'].items()}
 summary['comparison_1000_checkpoint']='1000-example continuation best update 150'
 p.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
