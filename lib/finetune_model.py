import argparse
import ast
import os
import random

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments)

INSTRUCTION = """Given a Hinglish code-mixed sentence, assign a Language Identification (LID) tag to each token.
Tag each word or word group with language labels: use 'h' for Hindi words, 'e' for English words, and 'u' for numbers, punctuation, and unidentified tokens. Keep punctuation attached to the preceding word and only break tokens at spaces. Return space-separated word-tag pairs.
Input: @virat ने Mumbai में match खेला 15 August को
Output: @virat u ने h Mumbai e में h match e खेला h 15 u August e को h"""

def load_examples(csv_path):
    examples = []
    for _, row in pd.read_csv(csv_path).iterrows():
        try:
            tags = ast.literal_eval(row["consolidated_LID_tags"])
            answer = " ".join(f"{tag['key']} {tag['value']}" for tag in tags)
            examples.append({"input": str(row["Sentences"]), "labels": answer})
        except (ValueError, SyntaxError, KeyError, TypeError):
            continue
    if not examples:
        raise ValueError("No valid examples found in the CSV file.")
    return examples

class LIDDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        prompt = f"Instruction: {INSTRUCTION}\nSentence: {example['input']}\nOutput: "
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        answer_ids = self.tokenizer(
            example["labels"] + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the downloaded model checkpoint")
    parser.add_argument("--data_csv", default="legacy/LID_with_consolidated.csv")
    parser.add_argument("--output_dir", default="finetuned_Model/lid_tagging")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    random.seed(42)
    examples = load_examples(args.data_csv)
    random.shuffle(examples)
    if len(examples) <= 200:
        raise ValueError("The legacy setup requires more than 200 examples for validation.")
    validation, training = examples[:200], examples[200:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.uint8,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.float16
        # torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=1,
        logging_steps=1,
        learning_rate=2e-4, weight_decay=0.01, warmup_ratio=0.1,
        max_grad_norm=1.0, lr_scheduler_type="cosine",
        fp16=True, seed=42,
        metric_for_best_model="eval_loss", load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=LIDDataset(training, tokenizer),
        eval_dataset=LIDDataset(validation, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
    )
    trainer.train()
    ouptut_file = os.path.join(args.output_dir, f"checkpoint-{args.model_path.split('/')[-1]}")
    trainer.save_model(ouptut_file)
    tokenizer.save_pretrained(ouptut_file)

if __name__ == "__main__":
    main()
