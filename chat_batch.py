"""Local batch generation and conservative, transparent rule-based grading."""
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time
import unicodedata

import torch


OBJECTIVE = {
    '日本の首都は？': (r'(?:日本の首都は)?東京(?:都)?(?:です|だ)?[。.!！]*', '東京'),
    '犬は動物ですか？': (r'(?:はい[、,。 ]*)?(?:犬は)?動物(?:です|だ)[。.!！]*|はい[。.!！]*', '犬が動物であることの肯定'),
    '1+1はいくつ？': (r'(?:1\+1(?:の答え)?は|答えは)?(?:2|二)(?:です|だ)?[。.!！]*', '2'),
    '3と5はどちらが大きい？': (r'(?:5|五)(?:(?:の方|のほう)?が大きい(?:です)?|です|だ)?[。.!！]*', '5が大きい'),
    '800円の30%引きはいくら？': (r'(?:(?:支払額|販売価格|値引き後(?:の価格)?|答え)は)?560円(?:です|だ)?[。.!！]*', '値引き後560円'),
    '14時から30分後は何時？': (r'(?:(?:時刻|答え)は)?14時30分(?:です|だ)?[。.!！]*', '14時30分'),
}
NATURAL = {
    'こんにちは': r'こんにちは[。!！]*',
    'おはよう': r'おはよう(?:ございます)?[。!！]*',
    'こんばんは': r'こんばんは[。!！]*',
    '最近どう？': r'(?:元気です|元気にしています|ぼちぼちです|変わりないです)[。!！]*',
    '元気ですか？': r'(?:はい[、。 ]*)?元気です[。!！]*',
    '眠い': r'(?:少し休んでください|ゆっくり休んでください|無理せず休んでください)[。!！]*',
    'ちょっと疲れた': r'(?:お疲れさまです|お疲れ様です|少し休んでください|ゆっくり休んでください)[。!！]*',
    'お腹がすいた': r'(?:何か食べましょう|何か食べてください)[。!！]*',
    'ありがとう': r'(?:どういたしまして|いえいえ|こちらこそ(?:ありがとう(?:ございます)?)?)[。!！]*',
    'どういたしまして': r'ありがとう(?:ございます)?[。!！]*',
    'ごめんなさい': r'(?:大丈夫です|気にしないでください|気にしないで)[。!！]*',
}


def grade(question, answer):
    # Normalization is used ONLY for grading; stored/generated answer is untouched.
    q = unicodedata.normalize('NFKC', question).strip()
    text = unicodedata.normalize('NFKC', answer.text).strip()
    key = next((k for k in OBJECTIVE if unicodedata.normalize('NFKC', k) == q), None)
    if answer.reason == 'error':
        return '要確認', '生成エラー。モデル品質の正誤とは区別。'
    if not text:
        return '不正解', '空回答。'
    if answer.reason == 'repetition' or re.search(r'(.{4,}?)\1\1', text):
        return '不正解', '異常な反復。'
    if text.rstrip('。.!！?？') == q.rstrip('。.!！?？') and question not in ('こんにちは', 'おはよう', 'こんばんは'):
        return '不正解', '質問・発言の復唱で、応答になっていない。'
    if key:
        pattern, expected = OBJECTIVE[key]
        if re.fullmatch(pattern, text):
            return '正解', f'回答全体が正解「{expected}」を表す。'
        return '不正解', f'正解は「{expected}」。回答全体がそれを表していない（数字の包含だけでは採点しない）。'
    if question in ('明日は何曜日？', '今日は暑いね'):
        return '要確認', '日時・天候などの文脈が必要なため自動採点しない。'
    if question in NATURAL and re.fullmatch(NATURAL[question], text):
        return '自然な回答', '質問に対応する簡潔な挨拶・会話表現。'
    if any(s in text for s in ('こちらのページ', 'と表示されます', 'のことです', 'とは何ですか')):
        return '不正解', '挨拶・日常会話への応答として無関係な説明または問い返し。'
    return '要確認', '会話の自然さをルールだけでは確定できないため、人による確認が必要。'


def run_batch(chat, questions, source, output, max_new_tokens, safe_print, load_seconds):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for number, question in enumerate(questions, 1):
        if chat.device.type == 'cuda':
            torch.cuda.synchronize(chat.device)
        t = time.perf_counter()
        try:
            answer = chat.answer(question, max_new_tokens)
            if chat.device.type == 'cuda':
                torch.cuda.synchronize(chat.device)
            seconds = time.perf_counter() - t
            verdict, reason = grade(question, answer)
            row = {'number': number, 'question': question, 'answer': answer.text,
                   'verdict': verdict, 'reason': reason, 'seconds': seconds,
                   'stop_reason': answer.reason, 'generated_tokens': len(answer.tokens),
                   'token_ids': answer.tokens, 'unicode_repaired': answer.unicode_repaired,
                   'error': answer.detail}
        except Exception as exc:
            row = {'number': number, 'question': question, 'answer': '', 'verdict': '要確認',
                   'reason': '入力・実行エラー。次の質問へ進む。', 'seconds': time.perf_counter()-t,
                   'stop_reason': 'error', 'error': f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        safe_print(f"\n{number}. 質問：{question}\n   AI：{row['answer']}\n   判定：{row['verdict']}\n   理由：{row['reason']}\n   時間：{row['seconds']:.3f}秒", flush=True)
    counts = Counter(r['verdict'] for r in rows)
    summary = {'total': len(rows), 'correct': counts['正解'], 'natural': counts['自然な回答'],
               'incorrect': counts['不正解'], 'needs_review': counts['要確認'],
               'inference_seconds': sum(r['seconds'] for r in rows),
               'batch_wall_seconds': time.perf_counter()-started, 'model_load_seconds': load_seconds}
    data = {'created_at': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(chat.checkpoint),
            'tokenizer_checkpoint': str(chat.tokenizer_checkpoint), 'device': str(chat.device),
            'source': str(source.resolve()), 'max_new_tokens': max_new_tokens, 'generation': 'greedy',
            'context': 128, 'eos_id': 3, 'independent_questions': True,
            'grading': 'conservative local rules; whole-answer matching for six objective questions; no external information',
            'summary': summary, 'results': rows}
    # ASCII JSON escaping preserves lone surrogates losslessly too.
    (output/'results.json').write_text(json.dumps(data, ensure_ascii=True, indent=2)+'\n', encoding='utf-8')
    def cell(value):
        text = str(value).encode('utf-8', 'backslashreplace').decode('utf-8')
        return html.escape(text).replace('|', '&#124;').replace('\n', '<br>')
    lines = ['# 自作日本語AI 60M 一括テスト', '', f"checkpoint: `{chat.checkpoint}`", '',
             '質問ごとに独立したgreedy生成。回答内容は変更せず記録（表の改行・特殊文字のみエスケープ）。',
             '正解は客観問題、自然な回答は会話で別集計。判断できない自然さ・外部情報は要確認。',
             '自動採点は限定した表現のルールであり、全表現の意味を理解する採点器ではありません。', '',
             '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
             '|No|質問|AIの実回答|判定|判定理由|時間（秒）|停止理由|',
             '|---|---|---|---|---|---:|---|']
    for r in rows:
        lines.append('|'+'|'.join(cell(v) for v in [r['number'], r['question'], r['answer'], r['verdict'], r['reason'], f"{r['seconds']:.6f}", r['stop_reason']])+'|')
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    safe_print(f"\n集計：正解 {summary['correct']}、自然な回答 {summary['natural']}、不正解 {summary['incorrect']}、要確認 {summary['needs_review']}\n合計推論時間：{summary['inference_seconds']:.3f}秒（モデル読込 {load_seconds:.3f}秒、別計測）\n保存先：{output}")
    return 1 if any(r['stop_reason'] == 'error' for r in rows) else 0
