import os

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

save_dir = "./models/exp1"
output_dir = "./outputs/exp1"


def start_finetune(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    data_dir: str,
    training_filename: str,
    validation_filename: str,
):
    print(f"Starting finetunes for {model_name}...")

    train_path = os.path.join(data_dir, training_filename)
    valid_path = os.path.join(data_dir, validation_filename)

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": valid_path
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
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
        # gradient_accumulation_steps=8
        # gradient_checkpointing=True,
        # fp16=True,
        learning_rate=learning_rate,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
