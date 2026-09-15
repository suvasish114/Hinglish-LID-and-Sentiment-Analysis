import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set this before any torch/transformers import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"  # Increased split size

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LID.utils as utils  
from types import SimpleNamespace
import random
from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import wandb  # Uncomment if you want wandb logging

# Clear GPU cache at the start
torch.cuda.empty_cache()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs('./logs', exist_ok=True)
logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

INSTRUCTION = {
    "LID_tagging": (
        """ You are an expert in Language Identification (LID) for Hinglish (Hindi-English code-mixed) text. Your task is to identify and classify tokens in the given sentence.
            Tag each word or word group in the following text with language labels.
                Rules:
                - Use 'hi' for Hindi words: (e.g., Mujhe, निर्माण, भारत, सुविधा, karna, hai, shala, वैश्विक, आरोग्य केंद्र, महाराष्ट्र)
                - Use 'en' for English words: (e.g., ऑफिस, इंडिया, इंटरनेशनल, हेरिटेज, बैंक, कमिटमेंट, Awesome, Culture, Lifestyle, Alliance, initiative, for, of, lockdown, Maharashtra)
                - Use 'ot' for numbers, punctuation, and unidentified tokens: (e.g., #Bollywood, #BJP,  @PMOIndia, @narendramodi, , . - : @ = & * + )
                - Keep punctuation attached to the preceding word. 
                - Only break tokens at spaces.
                    
                Instructions:
                1. Analyze each word in the sentence and identify the entity type for each word.
                2. Be precise and consistent with entity classification.
                3. Do not add any other extra suggestions.
                4. Format: Return space-separated word-tag pairs.
                Example Input: प्रधानमंत्री  नरेन्द्र  मोदी  डिजिटल  इंडिया  मिशन  को  आगे  बढ़ाने  के  लिए  पिछले  सप्ताह  Google  के  CEO  सुंदर  पिचाई  से  मुलाकात  की  थी ।
                Output: प्रधानमंत्री hi नरेन्द्र hi मोदी hi डिजिटल en इंडिया en मिशन en को hi आगे hi बढ़ाने hi के hi लिए hi पिछले hi सप्ताह hi Google en के hi CEO en सुंदर hi पिचाई hi से hi मुलाकात hi की hi थी hi । ot """)
}

class CustomDataset(Dataset):
    def __init__(self, data, args, tokenizer, max_length=256):  # Increased max_length for better utilization
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

        # Improved tokenization handling
        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True
        )
        
        target_encoding = self.tokenizer(
            labels + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True
        )

        input_ids = input_encoding['input_ids'] + target_encoding['input_ids']
        labels = [-100] * len(input_encoding['input_ids']) + target_encoding['input_ids']
        
        # Truncate if too long
        if len(input_ids) > self.max_length * 2:
            input_ids = input_ids[:self.max_length * 2]
            labels = labels[:self.max_length * 2]
        
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
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

    # Enhanced quantization config for better memory utilization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use fp16 for V100
        bnb_4bit_quant_storage=torch.uint8
    )

    # Load model with better memory utilization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # Use fp16 for V100
        device_map='auto',  # Let it automatically use available memory
        max_memory={torch.cuda.current_device(): "32GiB"}, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Improved LoRA config for better performance
    lora_config = LoraConfig(
        r=64,              
        lora_alpha=128,    # Increased alpha proportionally
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none",
        task_type="TOKEN_CLS",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    check_model_device(model)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    # Training setup
    args = SimpleNamespace(
        model_name="/data3/rajvee.sheth/aya-expanse-8b",
        data_dir="/data3/rajvee.sheth/LID",
        task="lid_class",
    )

    logger.info("Loading dataset...")
    dataset = utils.mix_data(args)
    if not dataset:
        raise ValueError("Failed to load dataset")
        
    random.shuffle(dataset)
    val_size = min(200, int(len(dataset) * 0.15))  # smaller of 200 or 15%
    train_data = dataset[val_size:]
    val_data = dataset[:val_size]

    utils.save_json(dataset, "./Data/LID.json")

    logger.info("Initializing Model and Tokenizer.")
    model, tokenizer = setup_model_and_tokenizer(args)
    logger.info("Creating custom training dataset.")
    train_dataset = CustomDataset(train_data, args, tokenizer)
    val_dataset = CustomDataset(val_data, args, tokenizer)
    
    # Clean model name for output directory
    model_name_clean = args.model_name.replace("/", "_").replace("-", "_")
    output_dir = f"./finetuned_Model/lid_class/{model_name_clean}"

    # Optimized batch size for full memory utilization
    if "Llama" in args.model_name or "llama" in args.model_name:
        batch_size = 4      # Increased batch size
        grad_acc_steps = 8  # Reduced accumulation steps
    elif "aya-expanse" in args.model_name:
        batch_size = 3
        grad_acc_steps = 12
    else:
        batch_size = 4
        grad_acc_steps = 8

    logger.info(f"Using batch_size={batch_size}, grad_acc_steps={grad_acc_steps}")
    logger.info(f"Effective batch size: {batch_size * grad_acc_steps}")

    # Training arguments - optimized for full memory usage
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                    # Keep 3 epochs 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        eval_strategy="steps",
        eval_steps=100,                       
        save_strategy="steps",
        save_steps=100,                        # Less frequent saves
        save_total_limit=-1,                   
        logging_dir="./logs",
        logging_steps=50,                      # More detailed logging
        learning_rate=2e-4,                    # Slightly increased learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=True,  
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        no_cuda=False,
        local_rank=-1,
        report_to="wandb",  
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        dataloader_pin_memory=False,           # Disable pin memory for more GPU memory
        remove_unused_columns=False,
        # Memory optimization
        dataloader_num_workers=0,              # No additional workers
    )

    logger.info(f"Training arguments configured. Output directory: {output_dir}.")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            padding=True, 
            pad_to_multiple_of=8, 
            return_tensors="pt",
            label_pad_token_id=-100,
        ),
    )

    # Print memory usage before training
    logger.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # Clear cache before training
    torch.cuda.empty_cache()

    logger.info("Training started.")
    trainer.train()

    # Print memory usage after training
    logger.info(f"GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # Plot the loss curve
    logs = trainer.state.log_history
    train_losses = [log['loss'] for log in logs if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in logs if 'eval_loss' in log]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss", marker='o')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    if eval_losses:
        plt.subplot(1, 2, 2)
        eval_steps = [log['step'] for log in logs if 'eval_loss' in log]
        plt.plot(eval_steps, eval_losses, label="Validation Loss", marker='s', color='orange')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("lid_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Clear cache after training
    torch.cuda.empty_cache()
    logger.info("Training process completed successfully!")

if __name__ == "__main__":
    main()