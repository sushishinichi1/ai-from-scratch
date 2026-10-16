# 自作日本語AI 60M v1.0

## 起動

```bash
cd /home/keita/ai-from-scratch
.venv/bin/python chat.py
```

日本語の質問を入力します。`exit` / `quit` / EOF / Ctrl+Cで終了します。
モデルは起動時に一度読み込み、質問ごとに独立して回答します。会話履歴は使用しません。

```bash
.venv/bin/python chat.py --once "日本の首都は？"
.venv/bin/python chat.py --device cuda --max-new-tokens 64
.venv/bin/python chat.py --device cpu --once "こんにちは"
.venv/bin/python chat.py --help
```

既定は `--device auto --max-new-tokens 64`。CUDAが利用可能ならGPU、なければCPUです。
明示的な`--device cuda`でCUDAが利用できない場合は、CPUを指定する案内を表示します。
この環境の既存`.venv`（PyTorch・NumPy）を使用します。外部APIへの通信はありません。

### 対話入力欄とシェルの区別

`あなた:` はAIへの質問欄です。ここに `cd`、`ls`、`.venv/bin/python ...` などを入力しても
シェルコマンドは実行されず、モデルへの質問として扱われます。
コマンドを実行するときは、先に `exit` または `quit` でchat.pyを終了し、
ターミナルのシェルプロンプトに戻ってから入力してください。

## 正式な使用checkpoint

`logs/japanese_base_v3/base48k_to96k_sft1100_20261010/sft/results.json`と
`best_checkpoint.json`が指定するbestは次のファイルです。

- SFT: `checkpoints/japanese_base_v3/base48k_to96k_sft1100_20261010/sft/update_150.pt`
- update: 150、Validation Loss: **1.9382515982085584**
- Base: `checkpoints/japanese_base_v3/base48k_to96k_sft1100_20261010/base/update_96000.pt`
- tokenizer: 上記Baseの`tokenizer_config`（自作Byte-BPE version3、語彙1024）
- Transformer: 自作`TinyGPT`、幅896・6層・8heads・context128

この実験には`best.pt`という名前のファイルは存在しません。正式なbestは上記update150です。
checkpointの複製・改名・書き換えはせず、既存metadataから直接読み込みます。
SFTにはtokenizer設定がないため、起動時に由来Baseから設定だけを取得します。
学習やtokenizerの再学習は行いません。ローカルの信頼済みcheckpointを使用する構成です。

## 生成と入力制限

学習時と同じ`<|user|>\n質問\n<|assistant|>\n`を使い、greedyで生成します。
終了token ID **3** で停止します。beamは使用しません。
promptと回答の合計が128tokenを超えないよう、生成上限をcontext残量で制限します。
質問だけでpromptが128token以上になる場合は拒否し、短くする案内を表示します。
質問や会話区切りを黙って切り捨てません。会話区切り文字列や特殊な制御文字の入力も拒否します。

UTF-8の有効性と残り生成予算でtokenを制限し、未完成の末尾byteは表示しません。
不正な内部byte列はその部分だけ `�` に置換し、未完成の末尾を除外した場合も案内します。
出力文字列にサロゲートが混入した場合はその文字だけ `\ud800` 等のエスケープ表記にし、
正常な日本語は変更しません。入力にサロゲートがある場合は意味を変えて処理せず再入力を案内します。
端末・パイプの文字コードはUTF-8を使用してください。
同じ4〜24tokenの区間が3回連続した場合は、文字境界で反復停止します。
空回答は空と表示し、上限・反復・エラーによる停止は別途案内します。
回答内容を固定文や外部モデルへ差し替える処理はありません。

`--once`の終了コード: 通常0、生成・読込エラー1、入力エラー2。
対話中の入力エラーや通常の生成例外は表示して次の質問を受け付けます。

## 20問の一括テスト

```bash
.venv/bin/python chat.py --batch tests/batch_questions.txt
```

モデルは1回だけ読み込み、各問を独立した質問として順にgreedy生成します。
`--device`と`--max-new-tokens`も併用できます。`--once`とは同時指定できません。
ターミナルに全回答・判定・理由・時間を表示し、`logs/chat_batch20/results.json`と
`logs/chat_batch20/report.md`に保存します。同じコマンドの再実行でこの2ファイルを更新します。
客観問題の正解数と会話の「自然な回答」は別集計です。日時・天候や判断困難な自然さは要確認とします。
採点は保守的なローカルルールで、モデルの回答を変更したり外部APIに送信したりしません。
各問の時間はCUDA同期込み、モデル読込時間は別計測です。

## Smoke test

```bash
.venv/bin/python tests/smoke_chat.py
.venv/bin/python tests/test_chat_unicode.py
```

実モデルで指定4問とEOS確認用の質問を処理します。CUDA/CPU、tokenizer、context制限、
同一モデル再利用、CLIの`--once`・`--help`・`exit`・`quit`を確認します。
空回答・例外回復だけはテスト内のstubで確認し、実アプリにはstubを組み込みません。
結果と実回答は`logs/chat_v1_unicode_smoke.json`へ保存します。正答率はテスト合否条件にしません。
UTF-8境界テストでは不完全な3/4byte文字、不正byte列、サロゲートを含む出力を模擬し、
厳格なUTF-8出力先でもCLIが継続することを確認します。

## 品質上の制限

短文QA用の小規模モデルです。計算・時刻・一般常識の誤答、質問の復唱、反復、内容不足があります。
アプリの正常動作は回答の正しさを保証しません。会話履歴・外部知識検索は未実装です。
既存の学習コード・dataset・checkpointは変更しません。
