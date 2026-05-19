import argparse
import os

from src.local_finetune import start_finetune
from src.tasks.reverse_experiments.reverse_task import REVERSE_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2-xl")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="june_version_7921032488")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_finetune(
        args.model_name,
        args.learning_rate,
        args.batch_size,
        args.n_epochs,
        os.path.join(REVERSE_DATA_DIR, args.dataset_name),
        "all_prompts_train.jsonl",
        "validation_prompts.jsonl",
    )