import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import time
import re

# Suppress unnecessary warnings
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configure GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Model configuration
checkpoint_path = "/data3/rajvee.sheth/LID/finetuned_Model/lid_class/aya_expanse_8b/checkpoint-1100"


# Load PEFT configuration
peft_config = PeftConfig.from_pretrained(checkpoint_path)

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load PEFT adapter
model = PeftModel.from_pretrained(model, checkpoint_path)

tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True,
    use_fast=True,  # Use fast tokenizer
    padding_side="left"  # Left padding for better batching
)
tokenizer.pad_token = tokenizer.eos_token

# Configure generation parameters for more accurate output
generation_config = {
    "max_new_tokens": 256,
    "do_sample": False,
    "num_beams": 4,       
    "temperature": 0.0,     # Pure greedy decoding
    "top_p": 1.0,          # No sampling
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True,
    "early_stopping": True,
    "length_penalty": 1.0,  # Prefer longer sequences
    "eos_token_id": tokenizer.eos_token_id,
    "repetition_penalty": 1.2,
    "output_scores": False, 
}

def generate_ner_tags_batch(texts):
    """Generation function with improved prompting for accuracy"""
    prompts = []
    for text in texts:

        prompt =  f"""You are an expert in Language Identification (LID) for Hinglish (Hindi-English code-mixed) text. Your task is to identify and classify tokens in the given sentence.
                      Tag each word or word group in the following text with language labels.
                        Rules:
                        - Use 'hi' for Hindi words: (e.g., Mujhe, निर्माण, भारत, सुविधा, karna, hai, shala, वैश्विक, आरोग्य केंद्र, महाराष्ट्र)
                        - Use 'en' for English words: (e.g., ट्रेन, आईपीएल, ट्विटर, ऑफिस, इंडिया, इंटरनेशनल, हेरिटेज, बैंक, कमिटमेंट, Awesome, Culture, Lifestyle, Alliance, initiative, for, of, lockdown, Maharashtra)
                        - Use 'ot' for numbers, punctuation, and unidentified tokens: (e.g., #Bollywood, #BJP,  @PMOIndia, @narendramodi, , . - : @ = & * + )
                        - Keep punctuation attached to the preceding word. 
                        - Only break tokens at spaces.
                            
                        Instructions:
                        1. Analyze each word in the sentence and identify the entity type for each word.
                        2. Be precise and consistent with entity classification.
                        3. Do not add any other extra suggestions.
                        4. Format: Return space-separated word-tag pairs.
                        Example Input: प्रधानमंत्री  नरेन्द्र  मोदी  डिजिटल  इंडिया  मिशन  को  आगे  बढ़ाने  के  लिए  पिछले  सप्ताह  Google  के  CEO  सुंदर  पिचाई  से  मुलाकात  की  थी ।
                        Output: प्रधानमंत्री hi नरेन्द्र hi मोदी hi डिजिटल en इंडिया en मिशन en को hi आगे hi बढ़ाने hi के hi लिए hi पिछले hi सप्ताह hi Google en के hi CEO en सुंदर hi पिचाई hi से hi मुलाकात hi की hi थी hi । ot
                        Process the given sentence:
                        Input: {text}
                        Output:"""

        prompts.append(prompt)

    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    print(inputs.input_ids.shape)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_config)

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Simply extract the output part after "Output:"
    results = [out.split('Output:')[-1].strip() for out in decoded_outputs]
    return results

def process_and_save_results(input_file, output_file, batch_size=8):
    """Process dataset and save results to CSV"""
    # Load and process data
    df = pd.read_csv(input_file)
    df = df.iloc[:]  # For testing, limit to first 10 rows
    
    sentences = df["Sentences"]
    # assert len(results) == valid_idx.sum(), f"Expected {valid_idx.sum()} results, got {len(results)}"
    results = []


    # Process in batches with progress bar
    with tqdm(total=len(sentences), desc="Processing") as pbar:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:min(i+batch_size, len(sentences))]
            try:
                batch_results = generate_ner_tags_batch(batch)
                results.extend(batch_results)
                pbar.update(len(batch))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    # Process one by one if batch fails
                    for sent in batch:
                        results.append(generate_ner_tags_batch([sent])[0])
                        pbar.update(1)
                else:
                    raise e
    
    # Update DataFrame and save
    df['predicted_lid_tags'] = pd.Series(results, index=df[df['Sentences'].notna()].index)
    df.to_csv(output_file, index=False)
    return df


# Main execution flow
if __name__ == "__main__":
    try:
        # File paths
        input_file = "/data3/rajvee.sheth/LID/LID_Test.csv"
        output_file = "/data3/rajvee.sheth/results/lid_aya_results.csv"
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process and save results
        print("Starting LID prediction process...")
        results_df = process_and_save_results(input_file, output_file)
        print(f"\nResults saved to: {output_file}")
        
        # Display sample results
        print("\nSample predictions:")
        for idx, row in results_df.head(5).iterrows():
            print(f"\nInput: {row['Sentences']}")
            print(f"Prediction: {row['predicted_lid_tags']}")
            
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        torch.cuda.empty_cache()
