import argparse
import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
import anthropic

def load_questions(question_file):
    with open(question_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    return questions

def generate_answer_openai(client, model, question, max_new_tokens, temperature=0):
    """
    OpenAI APIを使用して回答を生成する関数
    """
    messages = [
        {"role": "user", "content": question["turns"][0]}
    ]

    retries = 0
    max_retries = 5
    response = None
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                stop=None,
                temperature=temperature
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(5)

    if response is None:
        print(f"Failed to get answer for question_id: {question['question_id']}")
        return "Error: Failed to get answer."

    return response.choices[0].message.content.strip()

def generate_answer_claude(client, model, question, max_new_tokens, temperature=0):
    """
    Anthropic APIを使用して回答を生成する関数
    """
    messages = [
        {"role": "user", "content": question["turns"][0]}
    ]

    retries = 0
    max_retries = 5
    response = None
    while retries < max_retries:
        try:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(5)

    if response is None:
        print(f"Failed to get answer for question_id: {question['question_id']}")
        return "Error: Failed to get answer."
    
    return response.content[0].text

def main(args):
    # 質問データの読み込み
    questions = load_questions(args.question_file)

    # 結果の書き出し先ディレクトリを作成
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    
    # モデルに応じたクライアントと生成関数の選択
    if args.model.startswith("gpt"):
        client = OpenAI(api_key=args.api_key)
        generate_answer_func = generate_answer_openai
    elif args.model.startswith("claude"):
        client = anthropic.Anthropic(api_key=args.api_key)
        generate_answer_func = generate_answer_claude
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 各質問に対して回答を生成
    with open(args.answer_file, "w", encoding="utf-8") as f:
        for question in tqdm(questions):
            answer_text = generate_answer_func(client, args.model, question, args.max_new_tokens)

            result = {
                "question_id": question["question_id"],
                "answer_id": f"{question['question_id']}-{time.time()}",
                "model_id": args.model,
                "choices": [{"index": 0, "turns": [answer_text]}],
                "tstamp": time.time(),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID (e.g., gpt-4-1106-preview, claude-3-opus-20240229)")
    parser.add_argument("--question-file", type=str, default="data/mt_bench/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="data/mt_bench/model_answer/api_answers.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--api_key", type=str, help="API key")
    args = parser.parse_args()

    main(args)