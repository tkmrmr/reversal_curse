import argparse
import os

from src.local_finetune import start_finetune

INSTRUCTIONS_DIR = "data/instructions/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2-xl")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="copypaste_ug100_rg1000_main")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_finetune(
        args.model_name,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.n_epochs,
        os.path.join(INSTRUCTIONS_DIR, args.dataset_name),
        "all.jsonl",
        "unrealized_examples.jsonl",
        "exp3"
    )