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
    writer = open(answer_file, "w", encoding="utf-8")

    # 4. 質問ごとに推論
    for q in tqdm(questions, desc="Generating answers"):
        question_id = q["question_id"]
        # Mt-Benchの形式では複数ターン (turns) があり得るので注意
        # 例: q["turns"] = ["ユーザのターン1", "ユーザのターン2", ...]
        # ここでは turn を全てつなげたものをまとめて与える例を示す
        # （実際はturnごとに生成する方が望ましいが、簡易化のためにまとめて投げる）

        conversation_text = ""
        for i,turn_text in enumerate(q["turns"]):
            conversation_text += f"(ターン{i+1}): {turn_text}\n"

        # 5. vLLMで推論
        prompts = [conversation_text]
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        # vLLMの出力: outputs は List[vllm.Generation], かつ 1件のpromptにつき1件のGenerationがかえる
        generated_text = outputs[0].outputs[0].text  # best-of-1想定

        # 6. jmt-bench が期待する JSONL 形式に整形して書き出し
        # 参考: fastchat.llm_judge.gen_model_answer.get_model_answer() などの形式
        # "choices": [ { "turns": [...], ... } ] の形を作っておく
        # Mt-Bench は multi-turn なので turnごとに分割したいが、
        # 簡単化のため1ターン分として書く例
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

    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model repo id")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answer_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    generate_answers_vllm(
        model_name=args.model_name,
        question_file=args.question_file,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()
