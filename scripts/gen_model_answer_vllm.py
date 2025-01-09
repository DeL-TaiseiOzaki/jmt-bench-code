import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import sys
from config_singleton import WandbConfigSingleton
from fastchat.llm_judge.common import load_questions

def generate_answers_vllm(model_name, questions, answer_file, max_new_tokens=1024, temperature=0.1, top_p=1.0):
    """vLLM を用いて質問に対する回答を生成し、answer_file にJSONLで保存する。"""

    try:
        # 1. vLLMのモデルロード
        print(f"Loading model: {model_name}")
        llm = LLM(model=model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        sys.exit(1)

    # 3. JSONL書き込み準備
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    try:
        with open(answer_file, "w", encoding="utf-8") as writer:
            # 4. 質問ごとに推論
            for q in tqdm(questions, desc="Generating answers"):
                question_id = q.get("question_id", "unknown_id")
                turns = q.get("turns", [])
                conversation_text = ""
                for i, turn_text in enumerate(turns):
                    conversation_text += f"(ターン{i+1}): {turn_text}\n"

                # 5. vLLMで推論
                prompts = [conversation_text]
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                outputs = llm.generate(prompts, sampling_params)
                if not outputs:
                    generated_text = ""
                else:
                    generated_text = outputs[0].outputs[0].text.strip()

                # 6. jmt-bench が期待する JSONL 形式に整形して書き出し
                item = {
                    "question_id": question_id,
                    "model_id": model_name,
                    "choices": [
                        {
                            "turns": [generated_text]
                        }
                    ],
                }
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing to answer file {answer_file}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="vLLMを使ってモデル回答を生成します。")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFaceのモデルリポジトリIDまたはローカルパス")
    parser.add_argument("--answer_file", type=str, required=True, help="生成された回答を保存するJSONLファイルのパス")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="1回答あたりの最大生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.1, help="サンプリングの温度パラメータ")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-pフィルタリングの確率閾値")
    args = parser.parse_args()

    # WandbConfigSingleton から質問データを取得
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    if cfg.mtbench.bench_name and cfg.mtbench.dataset_dir:
        questions = load_questions(None, cfg.mtbench.bench_name, cfg.mtbench.dataset_dir)
    else:
        print("Error: bench_name or dataset_dir is not specified in the configuration.")
        sys.exit(1)

    generate_answers_vllm(
        model_name=args.model_name,
        questions=questions,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

if __name__ == "__main__":
    main()
