"""Clean the three Japanese wiki sources without training a model or tokenizer."""

import argparse
import bz2
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import re
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET

import mwparserfromhell
from mwparserfromhell import nodes


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'datasets/japanese_base_v2'
ORIGINAL_WIKIPEDIA = ROOT / 'datasets/japanese_wikipedia'
SOURCES = ('wikipedia', 'wikibooks', 'wikinews')
SOURCE_INFO = {
    'wikipedia': {
        'input': 'Existing 20231101.ja Wikipedia article files, unchanged',
        'license': 'CC-BY-SA-3.0 / GFDL, as listed for the existing dataset',
        'license_url': 'https://huggingface.co/datasets/wikimedia/wikipedia',
    },
    'wikibooks': {
        'input': 'jawikibooks-20261001-pages-articles-multistream.xml.bz2',
        'dump_url': 'https://dumps.wikimedia.org/jawikibooks/20261001/jawikibooks-20261001-pages-articles-multistream.xml.bz2',
        'license': 'CC-BY-SA-4.0',
        'license_url': 'https://ja.wikibooks.org/wiki/Wikibooks:著作権',
    },
    'wikinews': {
        'input': 'jawikinews-20261001-pages-articles-multistream.xml.bz2',
        'dump_url': 'https://dumps.wikimedia.org/jawikinews/20261001/jawikinews-20261001-pages-articles-multistream.xml.bz2',
        'license': 'CC-BY-4.0 (historical CC-BY-2.5-or-later)',
        'license_url': 'https://ja.wikinews.org/wiki/ウィキニュース:基本方針とガイドライン',
    },
}
JAPANESE = re.compile(r'[\u3041-\u30ff\u3400-\u9fff\uf900-\ufaff]')
URL = re.compile(r'(?:https?://|ftp://|www\.)[^\s<>\[\]]+', re.I)
SENTENCE = re.compile(r'[^。！？!?\n]+(?:[。！？!?]+[」』”"]*|$)', re.M)
MAGIC = re.compile(r'__[A-Z][A-Z0-9_]*__', re.I)
RESIDUAL = re.compile(r'\{\{|\}\}|\[\[|\]\]|\{\||\|\}|__[A-Z][A-Z0-9_]*__|</?[A-Za-z][^>]*>')
ADMIN_TITLE = re.compile(
    r'メインページ|サンドボックス|sandbox|編集(?:方法|の仕方|のしかた|練習)|'
    r'運営ページ|カテゴリ(?:ー)?一覧|カテゴリ索引|ナビゲーション|'
    r'ウィキ(?:ブックス|ニュース|ペディア)(?:について|へようこそ|の使い方)|'
    r'^(?:削除依頼|管理者への依頼|執筆依頼|投稿依頼|談話室|掲示板)(?:/|$)', re.I)
FOOTER_HEADING = re.compile(
    r'^(?:情報源|出典|参考文献|参考資料|脚注|注釈|外部リンク|関連(?:ニュース|記事|項目)|'
    r'カテゴリ(?:ー)?|目次|索引|書籍一覧|教科書(?:目録)?一覧|下位分野)$')
NAVIGATION = re.compile(
    r'前のページ|次のページ|目次(?:に|へ)(?:戻|移)|ナビゲーション|'
    r'(?:この記事|このページ|本項).{0,25}(?:編集|執筆).{0,20}(?:お願い|ください)|'
    r'ウィキ(?:ブックス|ニュース|ペディア).{0,20}(?:編集|執筆|読み方|使い方)|'
    r'^(?:姉妹プロジェクト|主要カテゴリ|編集方針|編集の仕方|蔵書一覧|進捗状況)')
CODE = re.compile(
    r'^\s*(?:def |class |import |from \w+ import|function\b|#include|'
    r'(?:SELECT|INSERT|CREATE TABLE)\b|(?:var|const|let) \w+\s*=|console\.|>>>|\$\s)|'
    r'(?:[{};]{3,})|\b(?:href|src|class|style)\s*=|'
    r'^\s*[A-Za-z_]\w*(?:\.\w+)*\s*\(.*\)\s*;?\s*(?://.*)?$|'
    r'^\s*(?:printf|fprintf|scanf|puts|print|println)\s*\(|'
    r'^\s*(?:std::|cout\s*<<|System\.out\.)|'
    r'^\s*return\s+(?:[\"\']|[^。]*;\s*(?://.*)?$)|'
    r'^\s*(?:if|for|while|switch)\s*\([^。]*\)\s*\{', re.I)
REMOVED_TAGS = {
    'ref', 'references', 'gallery', 'math', 'syntaxhighlight', 'source', 'pre',
    'code', 'script', 'style', 'timeline', 'score', 'imagemap', 'table',
}
ADMIN_PREFIX = re.compile(
    r'^(?:ファイル|画像|file|image|category|カテゴリ|カテゴリ―|template|テンプレート|'
    r'help|ヘルプ|special|特別|user|利用者|user talk|利用者‐会話|'
    r'wikibooks|wikinews|wikipedia|ウィキブックス|ウィキニュース|ウィキペディア|'
    r'portal|ポータル|mediawiki|commons|meta):', re.I)
LANGUAGE_PREFIX = re.compile(r'^[a-z]{2,3}(?:-[a-z0-9]+)?:', re.I)
INLINE_PROJECT = re.compile(r'^(?:w|b|n|s|wikt|wikipedia|wikibooks|wikinews|wikisource):', re.I)


def normalize(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def digest(text):
    return hashlib.blake2b(normalize(text).encode('utf-8'), digest_size=16).digest()


def sentence_keys(text):
    return {digest(match.group()) for match in SENTENCE.finditer(text)
            if len(normalize(match.group())) >= 20}


def remove_tables(text):
    """Also handle nested tables, which a single non-greedy regex cannot remove."""
    pieces, depth, start = [], 0, 0
    for match in re.finditer(r'\{\||\|\}', text):
        if match.group() == '{|':
            if depth == 0:
                pieces.append(text[start:match.start()])
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                start = match.end()
    if depth == 0:
        pieces.append(text[start:])
    return ''.join(pieces)


def remove_sections(text):
    lines, skip_level = [], None
    for line in text.splitlines():
        match = re.match(r'^\s*(={1,6})\s*(.*?)\s*\1\s*$', line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if FOOTER_HEADING.fullmatch(heading):
                skip_level = level
        elif line.strip() in {'参考文献', '脚注', '出典', '外部リンク', '関連項目'}:
            # The existing Wikipedia files already have plain-text headings.
            break
        if skip_level is None:
            lines.append(line)
    return '\n'.join(lines)


def wiki_to_text(raw):
    raw = html.unescape(raw).replace('\r\n', '\n').replace('\r', '\n')
    raw = re.sub(r'<!--.*?(?:-->|$)', '', raw, flags=re.S)
    raw = remove_sections(raw)
    raw = remove_tables(raw)
    # Standalone interwiki/navigation links have no useful article prose.
    raw = re.sub(r'(?m)^\s*\[\[\s*:?(?:[a-z][a-z-]{0,16}|特別|カテゴリ|Category):[^\n]*?\]\]\s*$', '', raw, flags=re.I)
    if any(marker in raw for marker in ('{{', '[[', '<', "''", '[http')):
        code = mwparserfromhell.parse(raw)
        raw = render_nodes(code)
    raw = MAGIC.sub('', raw)
    raw = re.sub(r'</?[A-Za-z][^>]*>', '', raw)
    return raw


def render_nodes(code):
    """Walk the parsed tree once; avoid a full-tree search for every replacement."""
    pieces = []
    for node in code.nodes:
        if isinstance(node, (nodes.Template, nodes.Argument, nodes.Comment)):
            continue
        if isinstance(node, nodes.Wikilink):
            target = str(node.title).strip().lstrip(':')
            if ADMIN_PREFIX.match(target) or LANGUAGE_PREFIX.match(target):
                continue
            if node.text is not None:
                pieces.append(render_nodes(node.text))
            else:
                pieces.append(INLINE_PROJECT.sub('', render_nodes(node.title)).split('#', 1)[0])
        elif isinstance(node, nodes.ExternalLink):
            if node.title is not None:
                pieces.append(render_nodes(node.title))
        elif isinstance(node, nodes.Tag):
            name = str(node.tag).strip().lower()
            if name in REMOVED_TAGS:
                continue
            if name in {'br', 'hr'}:
                pieces.append('\n')
            elif node.contents is not None:
                contents = render_nodes(node.contents)
                pieces.append('\n' + contents + '\n' if name in {'div', 'p', 'section', 'blockquote'} else contents)
        elif isinstance(node, nodes.Heading):
            pieces.append('\n' + render_nodes(node.title) + '\n')
        elif isinstance(node, nodes.HTMLEntity):
            pieces.append(html.unescape(str(node)))
        else:
            pieces.append(str(node))
    return ''.join(pieces)


def clean_text(raw):
    text = wiki_to_text(raw)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    lines, prose_characters = [], 0
    for original in text.splitlines():
        line = re.sub(r'^\s*[#*:;]+\s*', '', original).strip()
        line = re.sub(r'[ \t\u3000]+', ' ', line)
        if not line:
            if lines and lines[-1]:
                lines.append('')
            continue
        if '\ufffd' in line or re.search(r'(?:ã.|â€|(?:縺.|繧.){3,})', line):
            continue
        if RESIDUAL.search(line) or CODE.search(line) or NAVIGATION.search(line):
            continue
        # Navigation lists and table cells are not paragraphs.
        if re.match(r'^[|!]', line) or (' > ' in line and '。' not in line):
            continue
        urls = URL.findall(line)
        if urls:
            line = URL.sub('', line).strip(' []()（）,、:：')
            if len(line) < 20:
                continue
        if len(line) < 4 or len(JAPANESE.findall(line)) / len(line) < 0.25:
            continue
        if re.search(r'(.)\1{15,}', line) or re.fullmatch(r'[\W\d_]+', line):
            continue
        if len(line) < 20 and not JAPANESE.search(line):
            continue
        lines.append(line)
        if len(line) >= 20:
            prose_characters += len(line)
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    if len(text) < 200 or prose_characters < 200 or len(JAPANESE.findall(text)) < 100:
        return None, 'short_or_non_prose'
    if len(JAPANESE.findall(text)) / len(text) < 0.35:
        return None, 'non_japanese'
    if raw.count('{{') >= 4 and len(text) < 400 and len(text) / max(1, len(raw)) < 0.2:
        return None, 'template_dominated'
    # A navigation index can have many long labels but little actual prose.
    if len(re.findall(r'[。！？]', text)) < 2 and raw.count('[[') > 10:
        return None, 'navigation_index'
    assert not RESIDUAL.search(text)
    return text, None


def load_wikipedia(stats, exclusions):
    documents = []
    for split in ('validation', 'train'):
        with (ORIGINAL_WIKIPEDIA / f'{split}_articles.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                old = json.loads(line)
                stats['wikipedia']['input_documents'] += 1
                stats['wikipedia']['raw_characters'] += len(old['text'])
                if ADMIN_TITLE.search(old['title']):
                    reason, text = 'administrative_title', None
                else:
                    text, reason = clean_text(old['text'])
                if reason:
                    exclusions['wikipedia'][reason] += 1
                    continue
                documents.append({
                    'id': 'wikipedia:' + str(old['id']), 'source': 'wikipedia',
                    'title': old['title'], 'url': old['url'],
                    'history_url': old['url'] + '?action=history',
                    'group': 'wikipedia:' + str(old['id']), 'original_split': split,
                    'cleaned_text': text,
                })
    print(f'Wikipedia cleaned: {len(documents):,} documents', flush=True)
    return documents


def load_dump(project, source, stats, exclusions):
    documents = []
    path = OUTPUT / 'raw' / f'{project}-20261001-pages-articles-multistream.xml.bz2'
    with bz2.open(path, 'rb') as stream:
        for _, page in ET.iterparse(stream, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            ns = {'m': page.tag.split('}')[0][1:]}
            title = page.findtext('m:title', namespaces=ns) or ''
            raw = page.findtext('m:revision/m:text', namespaces=ns) or ''
            stats[source]['input_documents'] += 1
            stats[source]['raw_characters'] += len(raw)
            reason = None
            if page.findtext('m:ns', namespaces=ns) != '0':
                reason = 'non_article_namespace'
            elif page.find('m:redirect', ns) is not None:
                reason = 'redirect'
            elif ADMIN_TITLE.search(title):
                reason = 'administrative_title'
            if not reason:
                text, reason = clean_text(raw)
            if reason:
                exclusions[source][reason] += 1
            else:
                page_id = page.findtext('m:id', namespaces=ns)
                url = f'https://ja.{source}.org/wiki/' + quote(title.replace(' ', '_'), safe='/')
                documents.append({
                    'id': f'{source}:{page_id}', 'source': source, 'title': title,
                    'url': url, 'history_url': url + '?action=history',
                    'revision_id': page.findtext('m:revision/m:id', namespaces=ns),
                    'group': f'{source}:' + (title.split('/')[0] if source == 'wikibooks' else page_id),
                    'cleaned_text': text,
                })
            if stats[source]['input_documents'] % 5000 == 0:
                print(f'{source}: scanned {stats[source]["input_documents"]:,}; kept {len(documents):,}', flush=True)
            page.clear()
    return documents


def deduplicate(documents, exclusions):
    """Remove normalized duplicate documents, near copies and repeated sentences."""
    # Existing Wikipedia validation has priority over train and other sources.
    documents.sort(key=lambda d: (
        d.get('original_split') != 'validation',
        {'wikipedia': 0, 'wikinews': 1, 'wikibooks': 2}[d['source']], d['id']))
    seen_documents, seen_sentences = set(), set()
    feature_postings = defaultdict(list)
    previous_features, kept = [], []
    for document in documents:
        text = document['cleaned_text']
        key, features = digest(text), sentence_keys(text)
        if key in seen_documents:
            exclusions[document['source']]['exact_duplicate'] += 1
            continue
        seen_documents.add(key)
        candidates = Counter()
        for feature in features:
            postings = feature_postings.get(feature, ())
            if len(postings) <= 64:
                candidates.update(postings)
        near_duplicate = False
        for index, overlap in candidates.most_common(12):
            other_features = previous_features[index]
            union = len(features) + len(other_features) - overlap
            if overlap >= 2 and union and overlap / union >= 0.85:
                near_duplicate = True
                break
        if near_duplicate:
            exclusions[document['source']]['near_duplicate'] += 1
            continue
        local, lines = set(), []
        for line in text.splitlines():
            if not line:
                lines.append('')
                continue
            parts = []
            for match in SENTENCE.finditer(line):
                sentence = match.group()
                if len(normalize(sentence)) >= 20:
                    sentence_key = digest(sentence)
                    if sentence_key in seen_sentences or sentence_key in local:
                        continue
                    local.add(sentence_key)
                parts.append(sentence)
            if parts:
                lines.append(''.join(parts))
        cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
        if len(cleaned) < 200 or sum(len(line) for line in cleaned.splitlines() if len(line) >= 20) < 200:
            exclusions[document['source']]['short_after_sentence_dedup'] += 1
            continue
        document['cleaned_text'] = cleaned
        keys = sentence_keys(cleaned)
        seen_sentences.update(keys)
        feature_index = len(previous_features)
        previous_features.append(features)
        for feature in features:
            if len(feature_postings[feature]) <= 64:
                feature_postings[feature].append(feature_index)
        kept.append(document)
    return kept


def split_documents(documents):
    assignments = {}
    for source in ('wikibooks', 'wikinews'):
        groups = sorted({d['group'] for d in documents if d['source'] == source})
        random.Random(42).shuffle(groups)
        validation = set(groups[:max(1, len(groups) // 10)])
        assignments.update({group: 'validation' if group in validation else 'train' for group in groups})
    splits = {'train': [], 'validation': []}
    for document in documents:
        document['split'] = document.get('original_split') or assignments[document['group']]
        splits[document['split']].append(document)
    for split in splits:
        random.Random(42).shuffle(splits[split])
    for field in ('id', 'group'):
        assert not ({d[field] for d in splits['train']} & {d[field] for d in splits['validation']})
    # Even retained portions of a duplicate document must not leak across splits.
    validation_sentences = set().union(*(sentence_keys(d['cleaned_text']) for d in splits['validation']))
    assert all(not validation_sentences.intersection(sentence_keys(d['cleaned_text'])) for d in splits['train'])
    return splits


def save(documents, splits, stats, exclusions, replace_generated=False):
    cleaned_dir = OUTPUT / 'cleaned'
    cleaned_dir.mkdir(exist_ok=replace_generated)
    mode = 'w' if replace_generated else 'x'
    for source in SOURCES:
        selected = [d for d in documents if d['source'] == source]
        stats[source]['adopted_documents'] = len(selected)
        stats[source]['excluded_documents'] = stats[source]['input_documents'] - len(selected)
        stats[source]['cleaned_characters'] = sum(len(d['cleaned_text']) for d in selected)
        stats[source]['japanese_characters'] = sum(len(JAPANESE.findall(d['cleaned_text'])) for d in selected)
        stats[source]['exclusion_reasons'] = dict(exclusions[source])
        assert stats[source]['excluded_documents'] == sum(exclusions[source].values())
        with (cleaned_dir / f'{source}.jsonl').open(mode, encoding='utf-8') as stream:
            for document in sorted(selected, key=lambda d: d['id']):
                stream.write(json.dumps(document, ensure_ascii=False) + '\n')
    for split, selected in splits.items():
        with (OUTPUT / f'{split}_documents.jsonl').open(mode, encoding='utf-8') as stream:
            for document in selected:
                stream.write(json.dumps(document, ensure_ascii=False) + '\n')
        # Only cleaned prose enters these files, never the raw dump text.
        with (OUTPUT / f'{split}.txt').open(mode, encoding='utf-8', newline='\n') as stream:
            for document in selected:
                stream.write(document['cleaned_text'] + '\n\n')
    summary = {
        'seed': 42, 'sources': stats,
        'total_characters': sum(s['cleaned_characters'] for s in stats.values()),
        'total_japanese_characters': sum(s['japanese_characters'] for s in stats.values()),
        'adopted_documents': len(documents),
        'excluded_documents': sum(s['excluded_documents'] for s in stats.values()),
        'counting': 'Raw dump characters and excluded documents include all XML namespaces; cleaned characters are final document bodies, excluding inter-document separators. Japanese character count includes hiragana, katakana and kanji only.',
        'splits': {split: {'documents': len(docs), 'characters': sum(len(d['cleaned_text']) for d in docs)}
                   for split, docs in splits.items()},
        'split_policy': 'Seed 42; original Wikipedia article membership retained; Wikibooks chapters grouped by parent book; remaining source groups split 90/10.',
        'document_overlap': 0, 'group_overlap': 0,
        'normalized_sentence_overlap_min_20_characters': 0,
        'near_duplicate_method': 'NFKC and whitespace normalization; sentence-fingerprint Jaccard >= 0.85; repeated sentences of at least 20 normalized characters removed globally.',
        'model_training_performed': False, 'tokenizer_training_performed': False,
    }
    (OUTPUT / 'cleaning_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / 'sources.json').write_text(json.dumps(SOURCE_INFO, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main(replace_generated=False):
    if replace_generated and not (OUTPUT / 'cleaning_summary.json').exists():
        raise FileNotFoundError('Only this pipeline\'s generated corpus can be replaced')
    if not replace_generated and ((OUTPUT / 'cleaned').exists() or any((OUTPUT / f'{split}_documents.jsonl').exists() for split in ('train', 'validation'))):
        raise FileExistsError('Cleaned corpus already exists; refusing to overwrite it')
    stats = {source: Counter() for source in SOURCES}
    exclusions = {source: Counter() for source in SOURCES}
    documents = load_wikipedia(stats, exclusions)
    documents.extend(load_dump('jawikinews', 'wikinews', stats, exclusions))
    documents.extend(load_dump('jawikibooks', 'wikibooks', stats, exclusions))
    print('Removing duplicate documents and repeated prose...', flush=True)
    documents = deduplicate(documents, exclusions)
    splits = split_documents(documents)
    assert all(not RESIDUAL.search(d['cleaned_text']) for d in documents)
    save(documents, splits, stats, exclusions, replace_generated=replace_generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-generated', action='store_true', help='Rebuild only the generated Base v2 corpus; source data and checkpoints stay read-only')
    main(replace_generated=parser.parse_args().replace_generated)
