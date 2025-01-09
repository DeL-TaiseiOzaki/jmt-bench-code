# jmt-bench-code

## 概要

`jmt-bench-code` は、[Mt-Bench](https://github.com/EleutherAI/mt-bench) スイートを使用して言語モデルを評価するためのベンチマークツールです。このツールは [Weights & Biases (W&B)](https://wandb.ai/) と統合されており、実験のトラッキングを行います。また、高速な推論を実現するために [vLLM](https://github.com/vllm-project/vllm) を活用し、評価モデルとして `gpt-4o-mini` を使用して対象言語モデルのパフォーマンスを評価します。

## 目次

- [前提条件](#前提条件)
- [インストール](#インストール)
- [設定](#設定)
  - [config.yaml](#configyaml)
- [認証](#認証)
  - [Weights & Biases (W&B) ログイン](#weights--biases-wb-login)
  - [Hugging Face ログイン](#hugging-face-login)
- [vLLMを使ったモデル回答の生成](#vllmを使ったモデル回答の生成)
- [評価の実行](#評価の実行)
- [ログとモニタリング](#ログとモニタリング)
- [必要なパッケージ](#必要なパッケージ)
- [トラブルシューティング](#トラブルシューティング)
- [貢献方法](#貢献方法)
- [ライセンス](#ライセンス)
- [謝辞](#謝辞)

---

## 前提条件

`jmt-bench-code` をセットアップする前に、以下の項目が準備されていることを確認してください：

- **Python 3.8+**: システムにPythonがインストールされていること。 [こちら](https://www.python.org/downloads/) からダウンロード可能です。
- **CUDA (オプション)**: GPUによるモデル推論を利用する場合、CUDAがインストールされ適切に設定されていること。
- **Hugging Face アカウント**: Hugging Faceにホストされているモデルにアクセスするために必要です。
- **Weights & Biases (W&B) アカウント**: 実験のトラッキングとログのために必要です。

---

## インストール

1. **リポジトリのクローン**

   ```bash
   git clone https://github.com/your-username/jmt-bench-code.git
   cd jmt-bench-code
   ```

2. **仮想環境の作成（オプションで推奨）**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
   ```

3. **依存パッケージのインストール**

   `pip` を最新に更新してください：

   ```bash
   pip install --upgrade pip
   ```

   その後、必要なパッケージをインストールします：

   ```bash
   pip install -r requirements.txt
   ```

   **注意**: `torch` や GPU互換性に関連する問題が発生した場合、[PyTorch インストールガイド](https://pytorch.org/get-started/locally/) を参照し、システムに適した指示に従ってください。

---

## 設定

すべての設定は `config.yaml` ファイルで管理されています。このファイルは、ベンチマークプロセスの動作方法（モデルの選択、ログの設定、評価設定など）を決定します。

### config.yaml

以下は、各セクションの説明付きのサンプル `config.yaml` です。特定の用途に合わせてパラメータを調整してください。

```yaml
# config.yaml

wandb:
  log: True
  entity: "weblab-geniac1"  # W&Bのエンティティ/チーム名
  project: "leaderboard_test"  # W&Bのプロジェクト名
  run_name: 'YOUR_RUN_NAME'  # この実行の名前例: 'model_evaluation_1'

github_version: v2.0.0  # 記録用のバージョンタグ

testmode: false  # テスト用の小規模データセットを使用する場合はtrueに設定

api: false  # APIベースのモデルを使用する場合はtrue、ローカルモデルを使用する場合はfalse

model:
  use_wandb_artifacts: false
  artifacts_path: ""
  pretrained_model_name_or_path: 'OUTPUT_DIR'  # 対象モデルのローカルパスまたはHuggingFaceリポジトリID
  trust_remote_code: true
  device_map: "auto"
  load_in_8bit: false
  load_in_4bit: false

generator:
  do_sample: false
  num_beams: 1
  top_p: 1.0
  top_k: 0
  temperature: 0.1
  repetition_penalty: 1.0

tokenizer:
  use_wandb_artifacts: false
  artifacts_path: ""
  pretrained_model_name_or_path: "OUTPUT_DIR"  # トークナイザーのローカルパスまたはHuggingFaceリポジトリID
  use_fast: false

# llm-jp-eval 用の評価設定
max_seq_length: 2048
dataset_artifact: "wandb-japan/llm-leaderboard/jaster:v11"  # データセットのW&Bアーティファクト
dataset_dir: "/jaster/1.2.6/evaluation/test"
target_dataset: "all"  # 評価に含めるデータセット {all, jamp, janli, ...}
log_dir: "./logs"
torch_dtype: "bf16"  # PyTorchのデータ型 {fp16, bf16, fp32}
custom_prompt_template: null  # カスタムプロンプトテンプレートがある場合
custom_fewshots_template: null  # カスタムの少数ショットテンプレートがある場合

metainfo:
  basemodel_name: "YOUR_RUN_NAME"  # ベースモデルの名前
  model_type: "open llm"  # モデルのタイプ {open llm, commercial api}
  instruction_tuning_method: "None"  # インストラクションチューニングの方法
  instruction_tuning_data: ["None"]  # インストラクションチューニングに使用したデータ
  num_few_shots: 0
  llm-jp-eval-version: "1.1.0"

# Mt-Bench 評価設定
mtbench:
  question_artifacts_path: 'wandb-japan/llm-leaderboard/mtbench_ja_question:v0'  # 質問のW&Bアーティファクトパス
  referenceanswer_artifacts_path: 'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v0'  # 参照解答のW&Bアーティファクトパス
  judge_prompt_artifacts_path: 'wandb-japan/llm-leaderboard/mtbench_ja_prompt:v1'  # ジャッジプロンプトのW&Bアーティファクトパス
  bench_name: 'japanese_mt_bench'
  model_id: null  # モデルの一意識別子。nullの場合、自動生成
  question_begin: null
  question_end: null
  max_new_token: 1024
  num_choices: 1
  num_gpus_per_model: 1
  num_gpus_total: 1
  max_gpu_memory: null
  dtype: bfloat16  # データ型 {None, float32, float16, bfloat16}

  # 評価モデル設定
  judge_model: 'gpt-4o-mini'  # 評価に使用するモデル
  mode: 'single'  # 評価モード {single, pair, ...}
  baseline_model: null  # 比較対象のベースラインモデル
  parallel: 1  # 並列プロセス数
  first_n: null  # 評価する最初の質問数

  # 会話テンプレート設定
  custom_conv_template: true
  conv_name: "custom"
  conv_system_template: "{system_message}"
  conv_system_message: "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
  conv_roles: "('指示', '応答')"
  conv_sep: "\n\n### "
  conv_sep2: ""
  conv_stop_token_ids: "[2]"
  conv_stop_str: "###"
  conv_sep_style: "custom"
  conv_role_message_separator: ":\n"
  conv_role_only_separator: ":\n"
```

### 主な設定パラメータ

- **W&B 設定 (`wandb`)**:
  - `entity`: W&Bのチーム名またはユーザー名。
  - `project`: この実行がログされるW&Bのプロジェクト名。
  - `run_name`: この評価実行の一意な名前。

- **モデル設定 (`model`)**:
  - `pretrained_model_name_or_path`: ローカルモデルのパスまたはHuggingFaceのリポジトリID（例: `"meta-llama/Llama-2-7b-hf"`）。
  - `device_map`: デバイスの配置を指定。`"auto"` は利用可能なGPUを自動的に使用。

- **評価設定 (`mtbench`)**:
  - `judge_model`: 評価に使用するモデル（ここでは `'gpt-4o-mini'` を設定）。
  - `mode`: 評価モード（`'single'` は単一ターン評価）。
  - `parallel`: 評価の並列プロセス数。

- **vLLM 設定**:
  - 明示的には設定されていませんが、推論スクリプト内で `vllm` が最適に動作するように適切に設定されていることを確認してください。

---

## 認証

評価を実行する前に、W&BとHugging Faceに認証を行い、必要なリソースにアクセスできるようにします。

### Weights & Biases (W&B) ログイン

1. **コマンドラインからのログイン**

   ```bash
   wandb login
   ```

   - プロンプトが表示されたら、[W&Bのアカウントページ](https://wandb.ai/authorize)から取得したAPIキーを入力します。

2. **ログインの確認**

   ログインが成功したか確認するには、以下を実行します：

   ```bash
   wandb status
   ```

### Hugging Face ログイン

1. **Hugging Face CLI のインストール**

   まだインストールしていない場合は、以下を実行します：

   ```bash
   pip install huggingface_hub
   ```

2. **コマンドラインからのログイン**

   ```bash
   huggingface-cli login
   ```

   - プロンプトが表示されたら、[Hugging Faceのアクセス トークンページ](https://huggingface.co/settings/tokens)から取得したアクセストークンを入力します。

3. **ログインの確認**

   ログインが成功したか確認するには、以下を実行します：

   ```bash
   huggingface-cli whoami
   ```

   - あなたのHugging Faceのユーザー名が表示されれば成功です。

---

## vLLMを使ったモデル回答の生成

高速な推論を実現するために、vLLMを使用して対象モデルの回答を生成します。以下の手順に従ってください。

### ステップ1: 推論スクリプトの準備

`gen_model_answer_vllm.py` というカスタムスクリプトを用意します。このスクリプトはvLLMを使用して質問に対する回答を生成し、jmt-benchが期待するJSONL形式で保存します。

```python
# gen_model_answer_vllm.py

import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os

def generate_answers_vllm(model_name, question_file, answer_file, max_new_tokens=1024):
    """vLLM を用いて question_file の質問に対する回答を生成し、answer_file にJSONLで保存する。"""

    # 1. vLLMのモデルロード
    llm = LLM(model=model_name)  # CPU-only環境の場合は device_map 設定等が必要

    # 2. 質問をロード
    with open(question_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # 3. JSONL書き込み準備
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "w", encoding="utf-8") as writer:
        # 4. 質問ごとに推論
        for q in tqdm(questions, desc="Generating answers"):
            question_id = q["question_id"]
            # Mt-Benchの形式では複数ターン (turns) があり得るので注意
            conversation_text = ""
            for i, turn_text in enumerate(q["turns"]):
                conversation_text += f"(ターン{i+1}): {turn_text}\n"

            # 5. vLLMで推論
            prompts = [conversation_text]
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=1.0,
                max_tokens=max_new_tokens,
            )
            outputs = llm.generate(prompts, sampling_params)
            # vLLMの出力: outputs は List[vllm.Generation], かつ 1件のpromptにつき1件のGenerationが返る
            generated_text = outputs[0].outputs[0].text.strip()  # best-of-1想定

            # 6. jmt-bench が期待する JSONL 形式に整形して書き出し
            item = {
                "question_id": question_id,
                "model_id": model_name,   # jmt-bench 側で model_id として扱う
                "choices": [
                    {
                        "turns": [generated_text]  # ここに複数ターンを格納する場合もある
                    }
                ],
            }
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="vLLMを使ってモデル回答を生成します。")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFaceのモデルリポジトリIDまたはローカルパス")
    parser.add_argument("--question_file", type=str, required=True, help="質問が記載されたJSONLファイルのパス")
    parser.add_argument("--answer_file", type=str, required=True, help="生成された回答を保存するJSONLファイルのパス")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="1回答あたりの最大生成トークン数")
    args = parser.parse_args()

    generate_answers_vllm(
        model_name=args.model_name,
        question_file=args.question_file,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()
```

### ステップ2: 推論スクリプトの実行

以下のコマンドを実行して、vLLMを使って対象モデルの回答を生成します。

```bash
python gen_model_answer_vllm.py \
  --model_name "your-huggingface-repo/model-name" \
  --question_file "path/to/question.jsonl" \
  --answer_file "FastChat/fastchat/llm_judge/data/japanese_mt_bench/model_answer/YOUR_MODEL_ID.jsonl" \
  --max_new_tokens 1024
```

**パラメータ説明:**

- `--model_name`: HuggingFaceのモデルリポジトリID（例: `"meta-llama/Llama-2-7b-hf"`）またはローカルパス。
- `--question_file`: ベンチマーク質問が記載された `question.jsonl` ファイルのパス。
- `--answer_file`: 生成された回答を保存するJSONLファイルのパス。`mtbench_eval.py` の設定に合わせてディレクトリ構造を維持してください。
- `--max_new_tokens`: 1回答あたりの最大生成トークン数。必要に応じて調整してください。

**注意:**

- `answer_file` のパスは `mtbench_eval.py` が期待するディレクトリ構造と一致していることを確認してください。
- スクリプトはjmt-benchが期待するJSONL形式で回答を生成します。

---

## 評価の実行

モデルの回答が生成されたら、ベンチマークツールを使用してモデルを評価します。

### ステップ1: 評価スクリプトの準備

評価の主なスクリプトは `run_jmtbench_eval.py` です。このスクリプトは評価プロセス（データの読み込み、モデル推論、ジャッジメント、ログ記録）をオーケストレーションします。

```python
# run_jmtbench_eval.py

import wandb
import os
import sys
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from llm_jp_eval.evaluator import evaluate
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu

# 設定の読み込み
if os.path.exists("configs/config.yaml"):
    cfg = OmegaConf.load("configs/config.yaml")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
else:
    # config.yaml が存在しない場合のデフォルト設定
    cfg_dict = {
        'wandb': {
            'entity': 'default_entity',
            'project': 'default_project',
            'run_name': 'default_run_name'
        }
    }

# W&Bのセットアップとアーティファクト処理
wandb.login()
run = wandb.init(
    entity=cfg_dict['wandb']['entity'],
    project=cfg_dict['wandb']['project'],
    name=cfg_dict['wandb']['run_name'],
    config=cfg_dict,
    job_type="evaluation",
)

# WandbConfigSingleton の初期化
WandbConfigSingleton.initialize(run, wandb.Table(dataframe=pd.DataFrame()))
cfg = WandbConfigSingleton.get_instance().config

# 設定をアーティファクトとして保存
if cfg.wandb.log:
    if os.path.exists("configs/config.yaml"):
        artifact_config_path = "configs/config.yaml"
    else:
        # "configs/config.yaml" が存在しない場合、run.configの内容をYAMLとして書き出す
        instance = WandbConfigSingleton.get_instance()
        assert isinstance(instance.config, DictConfig), "instance.config must be a DictConfig"
        with open("configs/config.yaml", 'w') as f:
            f.write(OmegaConf.to_yaml(instance.config))
        artifact_config_path = "configs/config.yaml"

    artifact = wandb.Artifact('config', type='config')
    artifact.add_file(artifact_config_path)
    run.log_artifact(artifact)

# 評価フェーズ
# 1. llm-jp-eval 評価
# evaluate()
# cleanup_gpu()

# 2. Mt-Bench 評価
mtbench_evaluate()
# cleanup_gpu()

# W&Bへの結果ログ
if cfg.wandb.log and run is not None:
    instance = WandbConfigSingleton.get_instance()
    run.log({
        "leaderboard_table": instance.table
    })
    run.finish()
```

### ステップ2: 評価スクリプトの実行

以下のコマンドを実行して、ベンチマークプロセスを開始します。

```bash
python run_jmtbench_eval.py
```

**プロセス概要:**

1. **設定の読み込み**: `config.yaml` を読み込み、評価パラメータを設定。
2. **W&Bの初期化**: W&Bにログインし、新しい実行を初期化。
3. **アーティファクトのログ保存**: 現在の設定をW&Bアーティファクトとして保存し、再現性を確保。
4. **評価の実行**:
   - **llm-jp-eval**: （現在コメントアウトされている）追加の評価のプレースホルダー。
   - **Mt-Bench 評価**: `mtbench_evaluate()` を実行してコアベンチマークを実施。
5. **結果のログ記録**: 評価結果を集約し、W&Bにログとして記録。

**注意:**

- 前のステップで生成した `model_answer.jsonl` が `config.yaml` の設定に従って正しく配置されていることを確認してください。
- スクリプトは評価モデルとして `gpt-4o-mini` を使用するように設定されています。モデルがアクセス可能で適切に設定されていることを確認してください。

---

## ログとモニタリング

すべての評価実行はW&Bにログとして記録され、簡単にトラッキングと可視化が可能です。

- **Leaderboard Table**: 評価対象モデルのパフォーマンス指標を表示。
- **Radar Chart**: カテゴリごとのスコアを視覚的に表示。
- **Configuration Artifact**: 実行時の正確な設定を保存し、再現性を確保。

W&Bのダッシュボードにアクセスして、進行中および完了した評価をモニタリングしてください。

---

## 必要なパッケージ

以下のパッケージがインストールされていることを確認してください。便利のため、サンプルの `requirements.txt` を提供します。

### サンプル `requirements.txt`

```txt
# requirements.txt

# コア依存関係
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
tqdm>=4.60.0

# 評価と設定
fastchat>=0.2.21
omegaconf>=2.3.0

# 並列処理
ray>=2.6.0

# ロギングとアーティファクト管理
wandb>=0.15.0

# Google Generative AI API
google-generativeai>=0.1.0

# Hugging Face統合
huggingface_hub>=0.16.0

# vLLMによる高速推論
vllm>=0.1.7

# 追加のユーティリティ（必要に応じて）
fsspec>=2023.1.0
PyYAML>=6.0
```

**インストール方法:**

```bash
pip install -r requirements.txt
```

**注意**: システムや特定の要件に応じて、パッケージのバージョンを調整したり、追加の依存関係をインストールする必要がある場合があります。

---

## トラブルシューティング

### 1. vLLMのインストール問題

- **問題**: `vllm` に関連するインストールや実行時のエラー。
- **解決策**: 環境が [vLLMの前提条件](https://github.com/vllm-project/vllm#requirements) を満たしていることを確認してください。PythonやCUDAのバージョンが互換性があるか確認し、必要に応じて環境を調整してください。

### 2. Hugging Faceの認証エラー

- **問題**: プライベートモデルにアクセスできない、または認証に失敗する。
- **解決策**: `huggingface-cli login` を実行し、適切なアクセストークンを使用してログインしていることを確認してください。アクセストークンが必要な権限を持っているか確認してください。

### 3. W&Bのログ記録問題

- **問題**: 評価結果がW&Bのダッシュボードに表示されない。
- **解決策**: `wandb login` が成功していること、`config.yaml` の `entity` と `project` 名が正しいことを確認してください。また、インターネット接続が安定していることを確認してください。

### 4. モデル推論の失敗

- **問題**: モデルのロードや推論中にエラーが発生する。
- **解決策**:
  - `config.yaml` の `pretrained_model_name_or_path` が正しいことを確認。
  - 必要なモデルファイルがアクセス可能で壊れていないことを確認。
  - GPUを使用している場合、CUDAの設定やGPUメモリが十分であることを確認。

### 5. JSONLフォーマットの問題

- **問題**: 評価中にJSONLフォーマットに関するエラーが発生する。
- **解決策**: 生成された `model_answer.jsonl` が期待される構造になっていることを確認してください。以下のような形式である必要があります：

  ```json
  {
    "question_id": "unique_id",
    "model_id": "model_name",
    "choices": [
      {
        "turns": ["generated_answer"]
      }
    ]
  }
  ```

---

## 貢献方法

貢献は歓迎します！以下の手順に従ってください：

1. **リポジトリのフォーク**

2. **フィーチャーブランチの作成**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **変更内容のコミット**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **ブランチへのプッシュ**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **プルリクエストの作成**

   変更内容と解決した問題について明確に説明してください。

---

## ライセンス

このプロジェクトは [MITライセンス](LICENSE) の下でライセンスされています。

---

## 謝辞

- [Mt-Bench](https://github.com/EleutherAI/mt-bench) によるベンチマークスイートの提供。
- [vLLM](https://github.com/vllm-project/vllm) による高速推論機能。
- [Weights & Biases](https://wandb.ai/) による実験のトラッキングと可視化。
- [Hugging Face](https://huggingface.co/) による豊富な言語モデルのホスティング。

---

# クイックリファレンス

- **モデル回答の生成**:
  ```bash
  python gen_model_answer_vllm.py \
    --model_name "your-huggingface-repo/model-name" \
    --question_file "path/to/question.jsonl" \
    --answer_file "FastChat/fastchat/llm_judge/data/japanese_mt_bench/model_answer/YOUR_MODEL_ID.jsonl" \
    --max_new_tokens 1024
  ```

- **評価の実行**:
  ```bash
  python run_jmtbench_eval.py
  ```

- **W&Bへのログイン**:
  ```bash
  wandb login
  ```

- **Hugging Faceへのログイン**:
  ```bash
  huggingface-cli login
  ```

---

このガイドに従うことで、**vLLMを使用した高速な推論**を実現しつつ、`gpt-4o-mini` を用いた堅牢な評価により、`jmt-bench-code` を効果的にセットアップおよび実行できます。