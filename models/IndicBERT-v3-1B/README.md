---
language:
- en
- hi
- te
- ta
- bn
- ml
- mr
- kn
- gu
- as
- or
- pa
- sd
- ur
- brx
- doi
- kok
- ks
- mai
- mni
- ne
- sa
- sat
tags:
- indicbert
- gemma-3
- masked-next-token-prediction
- bidirectional
license: mit
datasets:
- ai4bharat/sangraha
- HuggingFaceFW/fineweb-2
- ai4bharat/IndicCorpV2
base_model:
- google/gemma-3-1b-it
pipeline_tag: token-classification
---

# IndicBERT-v3

IndicBERT-v3-1B is a multilingual, bidirectional encoder language model based on the **Gemma-3** architecture. Unlike standard causal LLMs, these models have been adapted to use **bidirectional attention**, making them highly effective for encoder-heavy tasks.

Also checkout [IndicBERT-v3-270M](https://huggingface.co/ai4bharat/IndicBERT-v3-270M) and [IndicBERT-v3-4B](https://huggingface.co/ai4bharat/IndicBERT-v3-4B).

## Model Description

- **Architecture:** Bidirectional Gemma-3 (Non-causal attention)
- **Vocabulary:** Standard Gemma-3 vocabulary
- **Objective:** Masked Next Token Prediction (MNTP)

## Training Strategy: Curriculum Learning

The models were trained using a rigid curriculum learning approach to balance English proficiency with Indic language adaptation while preventing catastrophic forgetting.

1.  **Phase 1 (English Foundation):** Continual pre-training on English text (ratio: 0.30).
2.  **Phase 2 (High/Mid-Resource Adaptation):** Adapted to 14 major Indic languages (Hindi, Telugu, Tamil, Bengali, Malayalam, Marathi, Kannada, Gujarati, Assamese, Oriya, Punjabi, Sindhi, Urdu, Nepali) with a 0.25 ratio.
3.  **Phase 3 (Low-Resource Generalization):** Introduction of 8 low-resource languages (Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Sanskrit, Santali) at a 0.15 ratio.
4.  **Phase 4 (Joint Consolidation):** The final 10% of training steps involved joint training on all 23 languages (0.25 ratio) to mitigate catastrophic forgetting.

## Training Data

The model was continually pre-trained on approximately **35 Billion tokens** (12M samples) sampled from various sources notably **Sangraha-Verified**, **FineWeb-2**, **IndicCorp-v2** amongst many other datasets. 
**Model was trained upto 4096 sequence length throughout the training.**

## ⚠️ Critical Warning: MNTP vs. MLM

This model was trained with **Masked Next Token Prediction (MNTP)**, not standard Masked Language Modeling (MLM like BERT).

- **BERT (MLM):** Mask token t_i; predict t_i using the hidden state at position i.
- **IndicBERT-v3 (MNTP):** Mask token t_i; predict t_i using the hidden state at position **i-1**.

**Implication:** When fine-tuning or using this model, you must ensure your data collation logic aligns with this shift.

## Note
Internal testing shows the models to be very strong when compared to existing Encoder LLMs. Dedicated text-encoder versions optimized for sentence embeddings and retrieval tasks will be released soon.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ai4bharat/IndicBERT-v3-1B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

text = "The capital of India is New Delhi."
target_word = " Delhi"

# 1. Tokenize
inputs = tokenizer(text, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids.clone()

# 2. Find target token index
# (Simple heuristic for demonstration)
target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
mask_idx = (input_ids[0] == target_token_id).nonzero(as_tuple=True)[0].item()

# 3. Mask the token
MASK_TOKEN_ID = tokenizer.mask_token_id  # Token ID of the mask is 4 
 
input_ids[0, mask_idx] = MASK_TOKEN_ID

# 4. Predict
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

# MNTP Rule: Prediction for token `i` comes from logits at `i-1`
pred_logits = logits[0, mask_idx - 1, :]
pred_token_id = torch.argmax(pred_logits).item()

print(f"Masked: {target_word}")
print(f"Predicted: {tokenizer.decode([pred_token_id])}")
```

## MNTP training

```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Configuration
MODEL_ID = "ai4bharat/IndicBERT-v3-1B"
DATA_PATH = "wikitext" # Example dataset
DATA_CONFIG = "wikitext-2-raw-v1"

class MNTPDataCollator:
    """
    Custom Data Collator for Masked Next Token Prediction.
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id  # Token ID of the mask is 4

    def __call__(self, examples):
        # 1. Create Batch
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = batch["input_ids"].clone()

        # 2. Create Mask
        # Create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        special_tokens_mask = torch.zeros(labels.shape, dtype=torch.bool)
        
        if self.tokenizer.all_special_ids:
            # We use a loop here which is safe and works across devices once we cast the result
            for special_id in self.tokenizer.all_special_ids:
                special_tokens_mask |= (labels == special_id)
        

        # Set probability to 0 for special tokens so they are never masked
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 3. Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 4. Apply Mask to Inputs
        # We replace the token at `i` with [MASK].
        # The label at `i` remains the original token (for prediction).
        input_ids[masked_indices] = self.mask_token_id

        labels[~masked_indices] = -100 # <- Comment this line out if you want to calculate loss on unmasked tokens too.

        # 5. Return Batch
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

def train():
    # 1. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. Load Dataset (Example: Wikitext)
    dataset = load_dataset(DATA_PATH, DATA_CONFIG, split="train[:1000]") # Small subset for demo
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. Setup Collator
    data_collator = MNTPDataCollator(tokenizer, mlm_probability=0.15)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./IndicBERT-v3-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        fp16=False,
        bf16=True, 
        save_strategy="epoch",
        remove_unused_columns=False, # Important for custom collators
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()
    
    # Save
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    train()
```

If you use this model, please consider citing:

```bibtex
@misc{indicbertv3,
  author = {Sidharth Pulipaka and Ashwin Sankar and Raj Dabre},
  title = {{IndicBERT-v3}: Multilingual Bidirectional Encoders for Indic Languages},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Repository},
  howpublished = {\url{https://huggingface.co/ai4bharat/IndicBERT-v3-1B}},
}
```