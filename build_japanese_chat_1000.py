"""Curate a context-bounded Japanese chat set and isolate train/validation/test."""

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from chat_data import format_prompt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'datasets/japanese_base_v2/raw/instruction_sources/GPT-4-Self-Instruct-Japanese.json'
OLD_DATA = ROOT / 'datasets/japanese_base_v2/chat_instruction_tiny.jsonl'
OUT = ROOT / 'datasets/japanese_base_v2/chat_1000'
TOKENIZER_PATH = ROOT / 'datasets/japanese_base_v2/bpe_1024.json'
SEED = 42
SOURCE_KEEP = 850
MAX_CONTEXT = 128

FIXED_TEST = [
    'アルゴリズムとは何ですか。',
    '天気と気候の違いは何ですか。',
    '友人が手伝ってくれました。何と伝えるとよいですか。',
    'なぜ海の水は塩辛いのですか。',
    'コンピュータのCPUは何をする部品ですか。',
]
NEW_TEST = [
    ('光合成では植物は何をしていますか。', '植物は光のエネルギーを使い、水と二酸化炭素から糖を作り、酸素を放出します。'),
    ('1,200円の品を25％引きで買うといくらですか。', '割引額は300円なので、支払額は900円です。'),
    ('水が沸騰する温度はいつも100℃ですか。', '標準大気圧では約100℃ですが、気圧が低い場所では沸点も下がります。'),
    ('「慎重」を使った短い文を作ってください。', '彼女は地図を見ながら、慎重に山道を進みました。'),
    ('急な予定変更を丁寧に知らせる文を作ってください。', '申し訳ありません。都合により、本日の予定を変更させていただけますでしょうか。'),
    ('海岸でごみを見つけたら、どう行動すればよいですか。', '安全に拾える物は持ち帰って分別し、危険物や大量のごみは自治体や管理者に知らせましょう。'),
    ('俳句の基本的な音数を教えてください。', '俳句は基本的に五・七・五の十七音で作ります。季語を含む形が伝統的です。'),
    ('日本の三大都市圏を挙げてください。', '一般に、首都圏・中京圏・近畿圏を指します。'),
    ('月が光って見えるのはなぜですか。', '月は太陽の光を反射しているため、地球から明るく見えます。'),
    ('CPUとメモリの役割を短く比べてください。', 'CPUは命令を処理し、メモリは作業中のデータや命令を一時的に置きます。'),
    ('「今日は雨だったので、試合は体育館で行われた。」を要約してください。', '雨のため、試合は体育館で開催されました。'),
    ('りんご5個を2人で同じ数ずつ分けるとどうなりますか。', '1人2個ずつで、1個余ります。'),
    ('友人に借りた本を返すときのお礼を一文で伝えてください。', '本を貸してくれてありがとう。とても楽しく読めました。'),
    ('歴史を調べるとき、複数の史料を比べる利点は何ですか。', '記録者や目的による違いを確かめ、出来事を一つの見方に偏らず考えやすくなります。'),
    ('駅まで徒歩12分です。9時30分に着くには何時に出ればよいですか。', '9時18分に出発すれば間に合います。'),
]
PARAPHRASE_TRAIN = [
    ('強風のため、船は港に戻った。', '風が強かったので、船は港へ引き返した。'),
    ('会議は参加者の都合により延期された。', '出席者の予定に合わせて、会議の日程が後ろに移された。'),
    ('この道は駅まで続いている。', 'この道を進むと駅に着きます。'),
    ('彼は毎朝、川沿いを走る。', '彼は朝になると、川のそばをランニングする。'),
    ('資料を読んでから、質問をまとめてください。', '資料に目を通した後、聞きたいことを整理してください。'),
    ('雨が止んだので、子どもたちは外で遊び始めた。', '雨上がりに、子どもたちは外へ出て遊びました。'),
    ('新しい制度は来月から始まる。', '新制度の開始は来月です。'),
    ('この植物は日当たりのよい場所を好む。', 'この植物は、よく日が当たる所で育ちやすい。'),
    ('説明が難しい場合は、具体例を加えるとよい。', '分かりにくい説明には、例を添えると伝わりやすくなります。'),
    ('列車の到着が遅れている。', '電車は予定時刻より遅れて着く見込みです。'),
    ('使い終わった道具は元の場所に戻してください。', '道具を使った後は、あった所へ片付けましょう。'),
    ('来場者の安全を確保するため、入口を一つにした。', '訪れた人が安全に過ごせるよう、出入り口を一か所にまとめた。'),
    ('気温が下がると、水は氷になる。', '水は冷やされて温度が下がると凍ります。'),
    ('彼女は地域の歴史を調べる活動に参加した。', '彼女は地元の歴史を探る取り組みに加わった。'),
    ('必要な情報を先に集めておこう。', 'まず、必要な資料を用意しておきましょう。'),
    ('この方法なら、作業にかかる時間を短くできる。', 'このやり方を使えば、作業時間を減らせます。'),
    ('図書館は静かに本を読める場所だ。', '図書館では落ち着いて読書ができます。'),
    ('結果を比べると、二つの条件に差が見られた。', '二つの条件を比較したところ、結果が異なっていた。'),
    ('申し込みの前に、日付と連絡先を確かめてください。', '申請する前に、日時と連絡先に誤りがないか確認しましょう。'),
    ('道路工事の影響で、バスは別の経路を通る。', '工事中はバスの運行ルートが変更されます。'),
    ('発表では、結論を最初に伝えると理解されやすい。', '発表の冒頭で結論を示すと、聞き手に内容が伝わりやすくなります。'),
    ('この地域では、春に多くの花が咲く。', '春になると、この土地のあちこちで花が開きます。'),
    ('利用者の意見を取り入れて、画面の配置を見直した。', '使う人の声を参考に、画面のレイアウトを変更した。'),
    ('期限に間に合わないと分かった時点で、担当者へ知らせよう。', '締め切りに遅れる可能性が出たら、すぐ担当の人に連絡しましょう。'),
    ('朝食を取ると、一日の活動に必要なエネルギーを補いやすい。', '朝ごはんは、日中に使う力を得る助けになります。'),
]
PARAPHRASE_FORMS = [
    '「{s}」を別の言葉で表してください。',
    '次の文を自然な日本語に言い換えてください。\n{s}',
    '意味を保って、読みやすく書き直してください：「{s}」',
    '「{s}」と同じ内容を、違う表現で伝えてください。',
    'この文を短く言い換えてください。\n{s}',
]
MISSING_FIXED_ANSWERS = {
    'なぜ海の水は塩辛いのですか。': '雨や川が岩石中の塩類を海へ運び、海水が蒸発しても塩類は残るためです。',
    'コンピュータのCPUは何をする部品ですか。': 'CPUはプログラムの命令を読み取り、計算や制御を行う部品です。',
}

DOMAIN_RULES = [
    ('daily_conversation', r'お礼|謝罪|依頼|挨拶|会話|メール|手紙|返事|伝え|相談|丁寧|マナー|接客|コミュニケーション'),
    ('computer', r'コンピュータ|コンピューター|プログラム|ソフトウェア|ハードウェア|CPU|メモリ|データ|ネット|インターネット|情報技術|アルゴリズム|コード|機械学習|AI|人工知能'),
    ('science', r'科学|物理|化学|生物|細胞|植物|動物|気候|地球|宇宙|水|熱|光|音|エネルギー|環境|実験|医学|健康|人体'),
    ('history', r'歴史|時代|戦争|文明|文化遺産|江戸|明治|昭和|古代|中世|近代|年表|歴史的'),
    ('geography', r'地理|地域|都道府県|県|市|町|島|山|川|海|国|大陸|気候|地形|人口|観光地|場所|地図'),
    ('music_culture', r'音楽|楽器|歌|芸術|美術|映画|文学|小説|詩|祭り|伝統|文化|演劇|絵画|舞踊|工芸'),
    ('math_basic', r'計算|数学|割合|％|パーセント|円|足し|引き|掛け|割り|数|確率|面積|長さ|時間|単位|図形|方程式'),
]
TASK_RULES = [
    ('summary', r'要約|まとめ|要点|主な内容'),
    ('paraphrase', r'言い換え|書き換え|別の表現|やさしい日本語|校正|文章を改善'),
    ('comparison', r'比較|違い|共通点|対照|比べ'),
    ('steps', r'手順|方法|やり方|順番|作り方|操作|進め方'),
    ('list', r'箇条書き|列挙|リスト|一覧|挙げて|いくつか'),
    ('reasoning', r'なぜ|どうして|理由|原因|根拠|場合|ときに|計算'),
    ('explanation', r'説明|とは|どのよう|何を|役割|仕組み'),
]


def norm(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'[\s\p{Punct}]', '', text) if False else re.sub(r'[^\wぁ-んァ-ヶ一-龠]', '', text)


def shingles(text, n=3):
    s = norm(text)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def domain_for(text):
    for label, pattern in DOMAIN_RULES:
        if re.search(pattern, text, re.I):
            return label
    return 'general_knowledge'


def task_for(text):
    for label, pattern in TASK_RULES:
        if re.search(pattern, text):
            return label
    if re.search(r'いくら|何個|何時|合計|計算', text):
        return 'math_basic'
    return 'general_knowledge'


def requested_count(question):
    m = re.search(r'([1-9])\s*(?:つ|項目|点|例|個)', question)
    return int(m.group(1)) if m else None


def compact_output(raw, question):
    text = raw.strip().replace('\r', '')
    text = re.sub(r'```.*?```', '', text, flags=re.S).strip()
    lines = [re.sub(r'^\s*(?:[-*•]|\d+[.)．、]|[・])\s*', '', x).strip() for x in text.splitlines() if x.strip()]
    explicit = requested_count(question)
    listish = len(lines) >= 2 and (len(lines) >= 3 or re.search(r'箇条書き|列挙|リスト|一覧|順番', question))
    if listish:
        if explicit and explicit > 4:
            return ''
        take = explicit or min(3, len(lines))
        if len(lines) < take:
            return ''
        text = '・' + '\n・'.join(lines[:take])
    else:
        text = re.sub(r'\n+', ' ', text)
        sentences = re.split(r'(?<=[。！？!?])\s*', text)
        # Keep one complete answer sentence; context 128 is the hard limit.
        text = sentences[0].strip()
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 320:
        return ''
    return text


def answer_candidates(raw, question):
    text = re.sub(r'```.*?```', '', str(raw), flags=re.S).replace('\r', '').strip()
    lines = [re.sub(r'^\s*(?:[-*•]|\d+[.)．、]|[・])\s*', '', x).strip()
             for x in text.splitlines() if x.strip()]
    list_lines = [x for x in lines if len(x) > 2 and not re.match(r'^(?:タイトル|結論|回答)[:：]', x)]
    explicit = requested_count(question)
    out = []
    if len(list_lines) >= 2 and (len(list_lines) >= 3 or re.search(r'箇条書き|列挙|リスト|一覧|順番', question)):
        want = explicit if explicit and explicit <= 3 else 2
        if explicit and explicit > 3:
            return []
        if len(list_lines) >= want:
            out.append('・' + '\n・'.join(list_lines[:want]))
        # Also consider complete first items if the pair is too long.
        out.extend('・' + line for line in list_lines[:min(len(list_lines), 8)])
    # Search later complete sentences too; first sentences are often long preambles.
    flat = re.sub(r'\n+', ' ', text)
    sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])\s*', flat) if s.strip()]
    for sentence in sentences[:12]:
        if re.match(r'^(?:もちろん|以下に|まず|ここでは|お答えします|ご質問ありがとうございます)', sentence):
            continue
        out.append(sentence)
    # Distinct, progressively shorter complete clauses are useful under context 128.
    for sentence in sentences[:6]:
        parts = [p.strip() for p in re.split(r'[、，,；;]', sentence) if p.strip()]
        for part in parts:
            if len(part) >= 8:
                out.append(part.rstrip('、，,；;') + '。')
    unique = []
    seen = set()
    for answer in out:
        answer = re.sub(r'\s+', ' ', answer).strip()
        key = norm(answer)
        if key and key not in seen and len(answer) <= 220:
            seen.add(key)
            unique.append(answer)
    return unique


def compact_question(instruction, context, raw_output, salt):
    task = task_for(instruction)
    body = instruction.strip()
    # Remove generated role-play framing while keeping the topic/task anchor.
    body = re.sub(r'^(?:タスク|指示|依頼)(?:\s*[:：]\s*)', '', body)
    topic = ''
    for delimiter in ('について', 'に関して', 'に関する', 'をテーマに', 'を題材に'):
        if delimiter in body:
            topic = body.split(delimiter, 1)[0]
            break
    if not topic:
        topic = re.split(r'[、。:：；;]', body, maxsplit=1)[0]
    topic = re.sub(r'^(?:次の|以下の|こちらの|この|その|与えられた)', '', topic).strip(' 「」『』\t ')
    topic = re.sub(r'^(?:内容|文章|テキスト|リスト|項目)(?:に|を|の|は|が).*$', '', topic).strip()
    if not topic:
        topic = re.sub(r'\s+', '', body)[:16]
    topic = topic[:28].rstrip('、。:：')

    variants = {
        'summary': [
            '「{t}」の要点を短くまとめてください。', '「{t}」を一文で要約してください。',
            '「{t}」について、中心となる内容を教えてください。', '「{t}」の概要を簡潔に説明してください。',
            '「{t}」を短く言い表してください。', '「{t}」から分かることをまとめてください。',
        ],
        'paraphrase': [
            '「{t}」を読みやすい表現に直してください。', '「{t}」をやさしい言葉で言い換えてください。',
            '「{t}」の意味を保って、別の表現にしてください。', '「{t}」を自然な日本語に整えてください。',
        ],
        'comparison': [
            '「{t}」の主な違いを簡潔に比べてください。', '「{t}」を比べ、異なる点を説明してください。',
            '「{t}」にはどんな共通点と違いがありますか。', '「{t}」の特徴を対比して教えてください。',
            '「{t}」を比べると、何が異なりますか。',
        ],
        'steps': [
            '「{t}」の基本的な手順を説明してください。', '「{t}」はどの順番で進めますか。',
            '「{t}」を行う方法を簡潔に教えてください。', '「{t}」の始め方を短く説明してください。',
        ],
        'list': [
            '「{t}」の例を二つ挙げてください。', '「{t}」の主な項目を簡潔に列挙してください。',
            '「{t}」には何がありますか。短く教えてください。', '「{t}」の代表例を挙げてください。',
        ],
        'reasoning': [
            '「{t}」となる理由を説明してください。', 'なぜ「{t}」なのでしょうか。',
            '「{t}」が起きる主な原因は何ですか。', '「{t}」の理由を簡潔に教えてください。',
            'どうして「{t}」と言えますか。',
        ],
        'explanation': [
            '「{t}」とは何ですか。簡潔に説明してください。', '「{t}」の概要を教えてください。',
            '「{t}」はどのようなものですか。', '「{t}」の主な特徴を短く説明してください。',
            '「{t}」について要点を教えてください。', '「{t}」の役割や仕組みを説明してください。',
        ],
        'general_knowledge': [
            '「{t}」について簡潔に教えてください。', '「{t}」の基本を説明してください。',
            '「{t}」とはどのようなものですか。', '「{t}」について知りたいです。要点を教えてください。',
            '「{t}」の特徴を一つ説明してください。', '「{t}」について短く答えてください。',
        ],
    }
    forms = variants.get(task, variants['general_knowledge'])
    template = forms[salt % len(forms)]
    if context and task == 'summary':
        source = re.split(r'(?<=[。！？!?])\s*', context.strip())[0]
        return f'次の内容を短く要約してください。\n{source}'
    return template.format(t=topic)


def make_row(q, a, ident, source, tokenizer, domain=None, task=None):
    q, a = q.strip(), a.strip()
    if not q or not a or '\ufffd' in q+a:
        return None
    if re.search(r'<\s*(?:div|script|table|code)|https?://|www\.', q+a, re.I):
        return None
    prompt = format_prompt(q)
    boundary_len = len(tokenizer.encode(prompt)) + len(tokenizer.encode(a))
    serialized = prompt + a + '\n'
    if boundary_len > MAX_CONTEXT or len(tokenizer.encode(serialized)) > MAX_CONTEXT:
        return None
    if tokenizer.decode(tokenizer.encode(serialized)) != serialized:
        return None
    return {
        'id': ident, 'category': domain or domain_for(q+' '+a),
        'task_type': task or task_for(q), 'source': source, 'question': q,
        'answer': a, 'text': serialized,
    }


def main():
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    old_rows = [json.loads(line) for line in OLD_DATA.read_text(encoding='utf-8').splitlines()]
    original = []
    fixed_by_question = {}
    for row in old_rows:
        expected = format_prompt(row['question']) + row['answer'] + '\n'
        if row.get('text') != expected:
            raise ValueError(f"Existing instruction format mismatch: {row.get('id')}")
        if (len(tokenizer.encode(format_prompt(row['question']))) + len(tokenizer.encode(row['answer'])) > MAX_CONTEXT
                or len(tokenizer.encode(expected)) > MAX_CONTEXT):
            raise ValueError(f"Existing instruction exceeds context 128: {row.get('id')}")
        r = dict(row)
        r['source'] = 'existing_handwritten_108'
        r['task_type'] = task_for(r['question'])
        r['category'] = r.get('category', domain_for(r['question']+' '+r['answer']))
        original.append(r)
        fixed_by_question[r['question']] = r
    fixed_rows = []
    for ix, question in enumerate(FIXED_TEST, 1):
        if question in fixed_by_question:
            fixed_rows.append(fixed_by_question[question])
        else:
            row = make_row(question, MISSING_FIXED_ANSWERS[question], f'ja-chat-fixed-test-{ix:02d}',
                           'handwritten_fixed_test', tokenizer)
            if row is None:
                raise ValueError(f'Fixed test example exceeds context: {question}')
            fixed_rows.append(row)

    new_test_rows = []
    for ix, (q, a) in enumerate(NEW_TEST, 1):
        row = make_row(q, a, f'ja-chat-test-{ix:02d}', 'handwritten_fixed_test', tokenizer)
        if row is None:
            raise ValueError(f'Fixed test example exceeds context or is malformed: {q}')
        new_test_rows.append(row)

    paraphrase_rows = []
    for ix, (sentence, answer) in enumerate(PARAPHRASE_TRAIN, 1):
        question = PARAPHRASE_FORMS[(ix - 1) % len(PARAPHRASE_FORMS)].format(s=sentence)
        row = make_row(question, answer, f'ja-chat-paraphrase-{ix:03d}', 'handwritten_paraphrase', tokenizer,
                       domain='language', task='paraphrase')
        if row is None:
            raise ValueError(f'Paraphrase example exceeds context 128: {question}')
        paraphrase_rows.append(row)

    fixed_prompts = FIXED_TEST + [q for q, _ in NEW_TEST]
    fixed_shingles = [shingles(q) for q in fixed_prompts]
    original_shingles = [shingles(x['question']) for x in original]
    raw = json.loads(SOURCE.read_text(encoding='utf-8'))
    rng = random.Random(SEED)
    rng.shuffle(raw)
    pools = defaultdict(list)
    short_rejected = Counter()
    for idx, item in enumerate(raw):
        instruction = str(item.get('instruction', '')).strip()
        context = str(item.get('input', '')).strip()
        original_q = instruction + ('\n' + context if context else '')
        raw_output = str(item.get('output', ''))
        q_options = [original_q, compact_question(instruction, context, raw_output, idx)]
        row = None
        for q in dict.fromkeys(q_options):
            for answer in answer_candidates(raw_output, q):
                row = make_row(q, answer, f'gpt4-si-{idx:05d}', 'CausalLM/GPT-4-Self-Instruct-Japanese', tokenizer)
                if row is not None:
                    break
            if row is not None:
                break
        if row is None:
            short_rejected['context_or_quality_filter'] += 1
            continue
        row['task_type'] = task_for(q)
        row['category'] = domain_for(q+' '+answer)
        qgrams = shingles(q)
        if any(jaccard(qgrams, tg) >= 0.48 for tg in fixed_shingles):
            short_rejected['fixed_test_overlap'] += 1
            continue
        if any(jaccard(qgrams, og) >= 0.78 for og in original_shingles):
            short_rejected['existing108_near_duplicate'] += 1
            continue
        pools[row['category']].append((row, qgrams))

    # Balanced subject selection: cap any single domain at 18%, then fill from
    # remaining valid examples. This preserves natural task diversity in source.
    selected = []
    selected_grams = []
    categories = sorted(pools)
    per_domain_cap = int(SOURCE_KEEP * 0.18)
    for category in categories:
        rng.shuffle(pools[category])
    progress = True
    while len(selected) < SOURCE_KEEP and progress:
        progress = False
        for category in categories:
            current = sum(1 for row in selected if row['category'] == category)
            if current >= per_domain_cap or not pools[category] or len(selected) >= SOURCE_KEEP:
                continue
            row, grams = pools[category].pop()
            if any(jaccard(grams, prior) >= 0.78 for prior in selected_grams):
                short_rejected['near_duplicate'] += 1
                continue
            selected.append(row)
            selected_grams.append(grams)
            progress = True
    if len(selected) < SOURCE_KEEP:
        rest = [pair for category in categories for pair in pools[category]]
        rng.shuffle(rest)
        for row, grams in rest:
            if len(selected) >= SOURCE_KEEP:
                break
            if any(jaccard(grams, prior) >= 0.78 for prior in selected_grams):
                short_rejected['near_duplicate'] += 1
                continue
            selected.append(row)
            selected_grams.append(grams)
    if len(selected) != SOURCE_KEEP:
        raise RuntimeError(f'Only {len(selected)} valid additional examples; need {SOURCE_KEEP}; '
                           f'filters={dict(short_rejected)}; candidates_by_domain='
                           f'{dict(Counter({k: len(v) for k, v in pools.items()}))}')

    # Deterministic splits. Test has 20 fixed evaluation examples plus 80 held-out
    # examples; validation receives 100; training receives the remaining 800.
    forced_test_ids = {x['id'] for x in fixed_rows}
    nonfixed_original = [x for x in original if x['id'] not in forced_test_ids]
    rng.shuffle(nonfixed_original)
    rng.shuffle(selected)
    val_original = []
    train_original = nonfixed_original
    source_test = selected[:80]
    source_val = selected[80:180]
    source_train = selected[180:]
    train_rows = train_original + source_train + paraphrase_rows
    val_rows = val_original + source_val
    test_rows = fixed_rows + new_test_rows + source_test
    if (len(train_rows), len(val_rows), len(test_rows)) != (800, 100, 100):
        raise RuntimeError(f'Unexpected split sizes: {len(train_rows)}, {len(val_rows)}, {len(test_rows)}')

    # Check all content fits, unique pairs/questions, and no near-duplicate question
    # crosses partitions. The 20 evaluation prompts are explicitly test-only.
    all_rows = train_rows + val_rows + test_rows
    pair_keys = [(norm(x['question']), norm(x['answer'])) for x in all_rows]
    if len(pair_keys) != len(set(pair_keys)):
        raise RuntimeError('Duplicate question/answer pair remains')
    test_keys = {norm(q) for q in fixed_prompts}
    if any(norm(x['question']) in test_keys for x in train_rows+val_rows):
        raise RuntimeError('A fixed test question leaked into train/validation')
    train_val_cross = []
    split_grams = {'train': [shingles(x['question']) for x in train_rows],
                   'validation': [shingles(x['question']) for x in val_rows],
                   'test': [shingles(x['question']) for x in test_rows]}
    for left_name, right_name in (('train','validation'),('train','test'),('validation','test')):
        for i, left in enumerate(split_grams[left_name]):
            for j, right in enumerate(split_grams[right_name]):
                if jaccard(left, right) >= 0.78:
                    train_val_cross.append([left_name, i, right_name, j])
                    if len(train_val_cross) > 20:
                        break
            if len(train_val_cross) > 20:
                break
    if train_val_cross:
        raise RuntimeError(f'Near-duplicate questions cross splits: {train_val_cross[:5]}')

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (('train', train_rows), ('validation', val_rows), ('test', test_rows)):
        (OUT / f'{name}.jsonl').write_text(
            ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')
    (OUT / 'fixed_test_questions.json').write_text(json.dumps(fixed_prompts, ensure_ascii=False, indent=2), encoding='utf-8')
    info = {
        'total_records': len(all_rows), 'train': len(train_rows), 'validation': len(val_rows), 'test': len(test_rows),
        'fixed_test_questions': len(fixed_prompts), 'source_added': len(selected),
        'handwritten_paraphrases': len(paraphrase_rows),
        'existing_108_retained': len(original), 'max_context': MAX_CONTEXT,
        'tokenizer_path': str(TOKENIZER_PATH), 'seed': SEED,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'near_duplicate_metric': 'Jaccard similarity over normalized character 3-gram sets; threshold 0.78',
        'cross_split_near_duplicate_count': len(train_val_cross),
        'exact_duplicate_count': len(all_rows)-len(set(pair_keys)),
        'domain_counts': dict(Counter(x['category'] for x in all_rows)),
        'task_type_counts': dict(Counter(x['task_type'] for x in all_rows)),
        'source_filters': dict(short_rejected),
        'source_attribution': {
            'name': 'GPT-4-Self-Instruct-Japanese', 'owner': 'CausalLM',
            'url': 'https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Japanese',
            'license': 'CC BY 4.0', 'revision': '9178a2d',
            'note': 'GPT-4 generated; source states it was filtered but not specially fact-verified; selected responses were compacted for context 128.'
        },
        'max_train_context_tokens': max(len(tokenizer.encode(format_prompt(x['question'])))+len(tokenizer.encode(x['answer'])) for x in train_rows),
        'max_validation_context_tokens': max(len(tokenizer.encode(format_prompt(x['question'])))+len(tokenizer.encode(x['answer'])) for x in val_rows),
        'max_test_context_tokens': max(len(tokenizer.encode(format_prompt(x['question'])))+len(tokenizer.encode(x['answer'])) for x in test_rows),
    }
    (OUT / 'dataset_report.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
