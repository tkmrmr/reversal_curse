import os
from typing import Literal

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import wandb
import torch


def start_finetune(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    n_epochs: int,
    data_dir: str,
    training_filename: str,
    validation_filename: str,
    experiment_name: Literal["exp1", "exp3"]
):
    print(f"Starting finetunes for {model_name}...")

    base_save_dir = f"./models/{experiment_name}"
    base_output_dir = f"./outputs/{experiment_name}"

    train_path = os.path.join(data_dir, training_filename)
    valid_path = os.path.join(data_dir, validation_filename)
    save_dir = os.path.join(base_save_dir, model_name.replace("/", "_"))
    output_dir = os.path.join(base_output_dir, model_name.replace("/", "_"))

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": valid_path
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16
    )
    
    # GPT-2系はpad_tokenがないためeos_tokenを代わりに使用
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        texts = [
            prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"])
        ]
        return tokenizer(
            texts,
            truncation=True, # max_lengthを超えるテキストを途中で切り詰める
            max_length=1024,  # GPT-2系の標準コンテキスト長
        )
    
    dataset = dataset.map(
        tokenize, 
        batched=True, # データをバッチ単位で処理
        remove_columns=dataset["train"].column_names
    )

    # batch化＆padding＆labelの作成
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, # 適用する場合は実効バッチサイズが同じになるようにbatch_sizeを小さくする
        # gradient_checkpointing=True, # 学習速度が20%遅くなる
        # fp16=True,
        learning_rate=learning_rate,
        optim="adamw_torch",
        weight_decay=0.0,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["wandb"],
    )

    wandb.init(
        project="reversal_curse",
        group=f"{experiment_name}",
        name=f"finetune_{model_name.replace('/', '_')}", 
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * training_args.gradient_accumulation_steps,
            "n_epochs": n_epochs,
            "fine_tuned_model": save_dir,
            "training_files": {
                "filename": train_path,
            },
            "data_path": data_dir,
        }
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    trainer.train()

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    wandb.finish()
