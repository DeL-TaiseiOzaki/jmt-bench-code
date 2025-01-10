#!/bin/bash

# Configuration variables
MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID="llama70"
NUM_GPUS_PER_MODEL=2
QUESTION_FILE="data/mt_bench/question.jsonl"
ANSWER_FILE="data/mt_bench/model_answer/${MODEL_ID}_answer.jsonl"
PROMPT_FILE="data/mt_bench/judge_prompts.jsonl"
JUDGMENT_FILE="data/mt_bench/model_judgment/${MODEL_ID}_judgment.jsonl"
OPENAI_API_MODEL="gpt-4o"
API_ANSWER_FILE="data/mt_bench/model_answer/gpt4o_answer.jsonl"
CLAUDE_API_MODEL="claude-3-opus-20240229"
CLAUDE_ANSWER_FILE="data/mt_bench/model_answer/${CLAUDE_API_MODEL}_answer.jsonl"

# Check if the required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY environment variable is not set."
  exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable is not set."
  exit 1
fi

# Create directories if they don't exist
mkdir -p $(dirname "$ANSWER_FILE")
mkdir -p $(dirname "$JUDGMENT_FILE")
mkdir -p $(dirname "$API_ANSWER_FILE")
mkdir -p $(dirname "$CLAUDE_ANSWER_FILE")

# Step 0: Generate model answers using OpenAI API
echo "Step 0: Generating model answers with OpenAI API..."
python gen_api_answer_claude.py \
    --model "$OPENAI_API_MODEL" \
    --question-file "$QUESTION_FILE" \
    --answer-file "$API_ANSWER_FILE" \
    --api_key "$OPENAI_API_KEY"

if [ $? -ne 0 ]; then
  echo "Error: API-based model answer generation failed."
  exit 1
fi

# Step 0.5: Generate model answers using Anthropic API
echo "Step 0.5: Generating model answers with Anthropic API..."
python gen_api_answer_claude.py \
    --model "$CLAUDE_API_MODEL" \
    --question-file "$QUESTION_FILE" \
    --answer-file "$CLAUDE_ANSWER_FILE" \
    --api_key "$ANTHROPIC_API_KEY"

if [ $? -ne 0 ]; then
  echo "Error: API-based model answer generation failed."
  exit 1
fi

# Step 1: Generate model answers using vLLM
echo "Step 1: Generating model answers with vLLM..."
python gen_model_answer.py \
    --model-path "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --num-gpus-per-model "$NUM_GPUS_PER_MODEL" \
    --question-file "$QUESTION_FILE" \
    --answer-file "$ANSWER_FILE"

if [ $? -ne 0 ]; then
  echo "Error: Model answer generation failed."
  exit 1
fi

# Step 2: Generate judgments using OpenAI API
echo "Step 2: Generating judgments with OpenAI API..."
python gen_judgments.py \
    --question-file "$QUESTION_FILE" \
    --answer-file "$ANSWER_FILE" \
    --prompt_file "$PROMPT_FILE" \
    --output-file "$JUDGMENT_FILE" \
    --model "$OPENAI_API_MODEL" \
    --api_key "$OPENAI_API_KEY"

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