## MT-Bench 評価スクリプト README

このリポジトリには、MT-Bench を用いてモデルの評価を行うためのスクリプトが含まれています。以下の手順で、モデルの回答生成、評価、スコア算出を自動的に実行できます。

**スクリプト概要:**

1.  **`gen_model_answer_vllm_single_turn.py`**: vLLM を使用して、Hugging Face モデルの回答を生成します。各質問の最初のターンのみの回答を生成します。
2.  **`gen_judgment_openai.py`**: OpenAI API を使用して、生成された回答を評価します。評価には、`data/mt_bench/judge_prompts.jsonl` に定義された日本語のプロンプトを使用します。
3.  **`calculate_score.py`**: 評価結果から平均スコアを算出します。
4.  **`run_evaluation.sh`**: 上記の 3 つのスクリプトを連続して実行し、評価プロセス全体を自動化します。

**ディレクトリ構成:**

```
.
├── calculate_score.py
├── gen_judgment_openai.py
├── gen_model_answer_vllm_single_turn.py
├── run_evaluation.sh
├── data
│   └── mt_bench
│       ├── question.jsonl
│       ├── judge_prompts.jsonl
│       ├── model_answer/
│       └── model_judgment/
└── README.md
```

*   `data/mt_bench/question.jsonl`: MT-Bench の質問セット。
*   `data/mt_bench/judge_prompts.jsonl`: 日本語の評価用プロンプトセット。
*   `data/mt_bench/model_answer/`: モデルの回答が保存されるディレクトリ。
*   `data/mt_bench/model_judgment/`: 評価結果が保存されるディレクトリ。

**前提条件:**

*   Python 3.7 以上
*   必要なパッケージ: `vllm`, `openai`, `numpy`, `tqdm`
    ```bash
    pip install vllm openai numpy tqdm
    ```
*   OpenAI API キー (環境変数 `OPENAI_API_KEY` に設定)

**使用方法:**

1.  **リポジトリのクローン:**
    ```bash
    git clone <リポジトリの URL>
    cd <リポジトリ名>
    ```

2.  **`run_evaluation.sh` の編集:**
    *   `MODEL_PATH` 変数に使用する Hugging Face モデルの ID を設定します。
    *   `MODEL_ID` 変数にモデルの任意の名前を設定します。
    *   必要に応じて、`NUM_GPUS_PER_MODEL`、`QUESTION_FILE`、`ANSWER_FILE`、`PROMPT_FILE`、`JUDGMENT_FILE`、`OPENAI_API_MODEL` を変更します。

    ```bash
    # Configuration variables
    MODEL_PATH="lmsys/vicuna-7b-v1.5"  # 使用したいモデルに変更
    MODEL_ID="vicuna-7b-v1.5"
    NUM_GPUS_PER_MODEL=1
    QUESTION_FILE="data/mt_bench/question.jsonl"
    ANSWER_FILE="data/mt_bench/model_answer/${MODEL_ID}_answer.jsonl"
    PROMPT_FILE="data/mt_bench/judge_prompts.jsonl"
    JUDGMENT_FILE="data/mt_bench/model_judgment/${MODEL_ID}_judgment.jsonl"
    OPENAI_API_MODEL="gpt-4-1106-preview" # 必要であれば変更
    ```

3.  **環境変数の設定:**
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

4.  **スクリプトの実行:**
    ```bash
    chmod +x run_evaluation.sh
    ./run_evaluation.sh
    ```

**実行結果:**

*   `data/mt_bench/model_answer/` ディレクトリにモデルの回答 (`.jsonl` 形式) が生成されます。
*   `data/mt_bench/model_judgment/` ディレクトリに評価結果 (`.jsonl` 形式) が生成されます。
*   コンソールに平均スコアが出力されます。

**カスタマイズ:**

*   **新しいモデルの評価:** `run_evaluation.sh` の `MODEL_PATH` と `MODEL_ID` を変更することで、異なるモデルを評価できます。
*   **プロンプトの変更:** `data/mt_bench/judge_prompts.jsonl` を編集することで、評価に使用するプロンプトを変更できます。
*   **質問セットの変更:** `data/mt_bench/question.jsonl` を置き換えることで、異なる質問セットを使用できます。
*   **パラメーター調整:** `gen_model_answer_vllm_single_turn.py` と `gen_judgment_openai.py` のコマンドライン引数を変更することで、vLLM や OpenAI API のパラメーターを調整できます。

**トラブルシューティング:**

*   `Error: OPENAI_API_KEY environment variable is not set.` というエラーが表示された場合は、環境変数 `OPENAI_API_KEY` が正しく設定されていることを確認してください。
*   その他のエラーが発生した場合は、各スクリプトのエラーメッセージを確認し、必要に応じてコードをデバッグしてください。

**注意事項:**

*   OpenAI API の使用には料金が発生します。`gpt-4-1106-preview` などの高機能モデルを使用する場合は特に注意してください。
*   `question.jsonl`には現在、最初の10問のみを格納しています。全問題で評価したい場合は、リポジトリのREADMEを参考に、完全な`question.jsonl`をダウンロードしてください。
*   本コードは学術的な目的で提供されており、商用利用には適さない場合があります。

この README が、MT-Bench 評価スクリプトの使用方法を理解するのに役立つことを願っています。
