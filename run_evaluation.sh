#!/bin/bash

# Configuration variables
MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"  #ここに使用したいモデルを記載
MODEL_ID="llama70"
NUM_GPUS_PER_MODEL=2
QUESTION_FILE="data/mt_bench/question.jsonl"
ANSWER_FILE="data/mt_bench/model_answer/${MODEL_ID}_answer.jsonl"
PROMPT_FILE="data/mt_bench/judge_prompts.jsonl"
JUDGMENT_FILE="data/mt_bench/model_judgment/${MODEL_ID}_judgment.jsonl"
OPENAI_API_MODEL="gpt-4o" 

# Check if the required environment variable is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY environment variable is not set."
  exit 1
fi

# Create the parent directory of ANSWER_FILE if it doesn't exist
ANSWER_DIR=$(dirname "$ANSWER_FILE")
if [ ! -d "$ANSWER_DIR" ]; then
  echo "Creating directory: $ANSWER_DIR"
  mkdir -p "$ANSWER_DIR"
fi

# Create the parent directory of ANSWER_FILE if it doesn't exist
JUDGMENT_DIR=$(dirname "$JUDGMENT_FILE")
if [ ! -d "$JUDGMENT_DIR" ]; then
  echo "Creating directory: $JUDGMENT_DIR"
  mkdir -p "$JUDGMENT_DIR"
fi

# # Step 1: Generate model answers using vLLM
# echo "Step 1: Generating model answers with vLLM..."
# python gen_model_answer.py \
#     --model-path "$MODEL_PATH" \
#     --model-id "$MODEL_ID" \
#     --num-gpus-per-model "$NUM_GPUS_PER_MODEL" \
#     --question-file "$QUESTION_FILE" \
#     --answer-file "$ANSWER_FILE"

if [ $? -ne 0 ]; then
  echo "Error: Model answer generation failed."
  exit 1
fi

# # Step 2: Generate judgments using OpenAI API
# echo "Step 2: Generating judgments with OpenAI API..."
# python gen_judgments.py \
#     --question_file "$QUESTION_FILE" \
#     --answer_file "$ANSWER_FILE" \
#     --prompt_file "$PROMPT_FILE" \
#     --output_file "$JUDGMENT_FILE" \
#     --model "$OPENAI_API_MODEL"

if [ $? -ne 0 ]; then
  echo "Error: Judgment generation failed."
  exit 1
fi

# Step 3: Calculate the average score
echo "Step 3: Calculating the average score..."
python get_score.py --judgment_file "$JUDGMENT_FILE"

if [ $? -ne 0 ]; then
  echo "Error: Score calculation failed."
  exit 1
fi

echo "Evaluation completed successfully!"