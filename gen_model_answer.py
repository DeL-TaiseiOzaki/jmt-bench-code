import argparse
import json
import time

import vllm
from vllm.sampling_params import SamplingParams

def main(args):
    # 1. JSONL（80行など）の読み込み
    with open(args.question_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. プロンプトとメタデータの準備
    prompts = []
    question_data = []

    for line in lines:
        q = json.loads(line)
        question_id = q["question_id"]
        category = q["category"]
        # turnsを単純に連結してプロンプトとして使用
        prompt = q["turns"][0]
        
        prompts.append(prompt)
        question_data.append({"question_id": question_id, "category": category})

    # 3. vLLMのLLMインスタンスを作成
    llm = vllm.LLM(
        model=args.model_path,
        tensor_parallel_size=args.num_gpus_per_model,
        max_model_len=1024
    )

    # 4. サンプリングパラメータの設定
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens
    )

    # 5. まとめて生成
    outputs = llm.generate(prompts, sampling_params)

    # 6. 結果をJSONL形式で書き出す
    with open(args.answer_file, "w", encoding="utf-8") as f:
        for output, qd in zip(outputs, question_data):
            choices = []
            for i, single_output in enumerate(output.outputs):
                choices.append({
                    "index": i,
                    "turns": [single_output.text.strip()]
                })

            result = {
                "question_id": qd["question_id"],
                "answer_id": f"{qd['question_id']}-{time.time()}",
                "model_id": args.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                      help="Hugging Face model ID or local path for the vLLM-compatible model")
    parser.add_argument("--model-id", type=str, required=True,
                      help="Model name to record in the output JSONL")
    parser.add_argument("--question-file", type=str, default="data/mt_bench/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="data/mt_bench/model_answer/answers.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1,
                      help="Number of parallel answers to generate for each question")
    parser.add_argument("--num-gpus-per-model", type=int, default=1,
                      help="Number of GPUs to use for tensor-parallel in vLLM")
    args = parser.parse_args()

    main(args)