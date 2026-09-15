import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # Set this before any torch/transformers import

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LID.utils as utils
import argparse
import random
from loguru import logger
import torch
import numpy as np
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.cuda.empty_cache()  # Clear GPU cache

random.seed(42)

os.makedirs('./logs', exist_ok=True)
logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

INSTRUCTION = {
    "LID_tagging": ( """Given a Hinglish code-mixed sentence, assign a Language Identification (LID) tag to each token.

                Tag each word or word group in the following text with language labels.
                Rules:
                - Use 'h' for Hindi words
                - Use 'e' for English words
                - Use 'u' for numbers, punctuation, and unidentified tokens
                - Keep punctuation attached to the preceding word
                - Only break tokens at spaces
                Format: Return space-separated word-tag pairs.
                Input: "@virat ने Mumbai में match खेला 15 August को`"
                Output: "@virat u ने h Mumbai e में h match e खेला h 15 u August e को h""")
}

class CustomDataset(Dataset):
    def __init__(self, data, args, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = "LID_tagging"
        self.instruction = INSTRUCTION[self.task]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['input']
        labels = item['labels']
        
        prompt = f"Instruction: {self.instruction}\nSentence: {sentence}\nOutput: "

        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None
        )

        target_encoding = self.tokenizer(
            labels + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None
        )

        label_ids = np.array([-100] * len(input_encoding['input_ids']) + target_encoding['input_ids'], dtype=np.int32)
        input_ids = np.array(input_encoding['input_ids'] + target_encoding['input_ids'], dtype=np.int32)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def check_model_device(model):
    devices = set()
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            devices.add(param.device.index)
    if len(devices) == 1:
        logger.info(f"All model parameters are on GPU {list(devices)[0]}")
    else:
        logger.warning(f"Model parameters are spread across GPUs: {devices}")

def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        max_memory={0: "24GiB"},
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "uint8"
        },
        offload_folder="offload",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    check_model_device(model)
    model.print_trainable_parameters()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for LID tagging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to fine-tune"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/himanshubeniwal/CodeMixing",
        help="Directory containing LID_Train.csv and LID_Test.csv files"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lid_tagging",
        choices=["lid_tagging"],
        help="Task type (only LID tagging supported)"
    )
   
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        parser.error(str(e))
        sys.exit(1)

    dataset = utils.mix_data(args)
    if not dataset or len(dataset) == 0:
        logger.error(
            f"No data loaded. Please check that '{os.path.join(args.data_dir, 'LID_Train.csv')}' exists and is not empty.\n"
            f"Current working directory: {os.getcwd()}\n"
            f"args.data_dir: {args.data_dir}\n"
            f"Full expected path: {os.path.join(args.data_dir, 'LID_Train.csv')}"
        )
        sys.exit(1)
    random.shuffle(dataset)
    val_size = 200
    if len(dataset) <= val_size:
        logger.error(f"Not enough data for training/validation split. Dataset size: {len(dataset)}, val_size: {val_size}")
        sys.exit(1)
    train_data = dataset[val_size:]
    val_data = dataset[:val_size]

    # if local_rank == 0:
    utils.save_json(dataset, "./Data/lid.json")

    logger.info("Initializing Model and Tokenizer.")
    model, tokenizer = setup_model_and_tokenizer(args)
    logger.info("Creating custom training dataset.")
    train_dataset = CustomDataset(train_data, args, tokenizer)
    val_dataset = CustomDataset(val_data, args, tokenizer)
    output_dir = f"./finetuned_Model/lid_tagging/{args.model_name}"

    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        batch_size = 8
    elif args.model_name == "CohereForAI/aya-expanse-8b":
        batch_size = 8
    elif args.model_name == "google/gemma-7b-it":
        batch_size = 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        eval_strategy="steps",  
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=1,
        logging_first_step=True,
        log_level='debug',
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=True,
        fp16_opt_level="O1",
        ddp_find_unused_parameters=False,
        seed=42,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        no_cuda=False,
        local_rank=-1,
    )

    logger.info(f"Training arguments configured. Output directory: {output_dir}.")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        padding=True, 
        pad_to_multiple_of=8, 
        return_tensors="pt"
    )

    logger.info("Initializing Trainer with model, tokenizer, and training dataset.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Training started.")
    trainer.train()

    logger.info("Training process completed.")

if __name__ == "__main__":
    main()