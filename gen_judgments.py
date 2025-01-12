import argparse
import json
import time
from tqdm import tqdm
from openai import OpenAI

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(args):
    # OpenAI クライアントを初期化（api_key を渡す）
    client = OpenAI()

    # 各種 JSONL ファイルを読み込み
    questions = load_jsonl(args.question_file)
    answers = load_jsonl(args.answer_file)
    prompts = load_jsonl(args.prompt_file)

    # 質問IDをキーにした辞書
    question_dict = {q["question_id"]: q for q in questions}
    # prompt名をキーにした辞書 (single-v1 / single-math-v1 など)
    prompt_dict = {p["name"]: p for p in prompts}

    with open(args.output_file, "w", encoding="utf-8") as f:
        for answer in tqdm(answers):
            question_id = answer["question_id"]
            category = question_dict[question_id]["category"]

            # ユーザの質問とモデル出力を取得
            question = question_dict[question_id]["turns"][0]
            answer_text = answer["choices"][0]["turns"][0]

            
            prompt_template = prompt_dict["single-v1"]["prompt_template"]
            prompt = prompt_template.format(
                question=question,
                answer=answer_text
            )

            # messages の先頭には single-v1 の system_prompt を入れる例
            messages = [
                {"role": "system", "content": prompt_dict["single-v1"]["system_prompt"]},
                {"role": "user",   "content": prompt}
            ]

            retries = 0
            max_retries = 5
            judgment = None

            while retries < max_retries:
                try:
                    # OpenAI クライアント経由でチャット補完 API を呼び出し
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        stop=None,
                        temperature=args.temperature
                    )
                    judgment = response.choices[0].message.content
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    retries += 1
                    time.sleep(5)

            if judgment is None:
                print(f"Failed to get judgment for question_id: {question_id}")
                judgment = "Error: Failed to get judgment."

            result = {
                "question_id": question_id,
                "model_id": answer["model_id"],
                "judgment": judgment,
                "tstamp": time.time(),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="data/mt_bench/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="data/mt_bench/model_answer/temp_answer.jsonl")
    parser.add_argument("--prompt-file", type=str, default="data/mt_bench/judge_prompts.jsonl")
    parser.add_argument("--output-file", type=str, default="data/mt_bench/model_judgment/temp_judgment.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview", help="OpenAI model to use for judgment")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for the generated judgment")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    args = parser.parse_args()

    main(args)
