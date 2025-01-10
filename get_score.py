import argparse
import json
import numpy as np
import re

def calculate_average_score(judgment_file):
    """
    Calculates the average score from a file containing judgments in JSON format (jsonl).
    Handles Japanese text and extracts numerical scores.

    Args:
        judgment_file: Path to the file containing judgments, one per line.

    Returns:
        The average score as a float, or None if no valid judgments are found.
    """
    judgments = []
    with open(judgment_file, "r", encoding="utf-8") as f:  # Specify UTF-8 encoding
        for line in f:
            try:
                judgment_data = json.loads(line)
                judgment_text = judgment_data["judgment"]

                # Extract score from judgment text using regular expressions
                match = re.search(r"(\d+(?:\.\d+)?)", judgment_text[-12:])  # Find numbers (including decimals)
                if match:
                    score = float(match.group(1))
                    judgments.append(score)
                else:
                    print(f"Warning: Could not parse score from judgment: '{judgment_text}'")

            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON format in line: '{line.strip()}'")

    if not judgments:
        print("Error: No valid judgments found.")
        return None

    average_score = np.mean(judgments)
    return average_score

def main(args):
    """
    Parses command-line arguments and calls calculate_average_score.
    """
    average_score = calculate_average_score(args.judgment_file)

    if average_score is not None:
        print(f"Average score: {average_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average score from a judgment file.")
    parser.add_argument("--judgment_file", type=str, required=True, help="Path to the judgment file.")
    args = parser.parse_args()
    main(args)