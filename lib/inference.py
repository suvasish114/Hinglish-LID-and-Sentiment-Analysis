# imports
from tqdm import tqdm
import csv, torch, os, sys
import pandas as pd
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# file specific configurations
prompt_base_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts"
zero_shot_prompt_path = Path(prompt_base_path).joinpath("zero_shot_lid_prompt.txt")
one_shot_prompt_path = Path(prompt_base_path).joinpath("one_shot_lid_prompt.txt")
prompts = [zero_shot_prompt_path, one_shot_prompt_path]

# BASELINE CLASS INITIALIZATION
class BaseModel:
    def __init__(self, zero_shot_prompt_path: str, one_shot_prompt_path: str, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        self.zero_shot_prompt_path = zero_shot_prompt_path
        self.one_shot_prompt_path = one_shot_prompt_path
        self.input_csv_file_path = input_csv_file_path
        self.output_csv_file_path = output_csv_file_path
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_prompt(self, text: str, shot: int) -> str: # return the prompt based on the shot value
        prompt_path = self.zero_shot_prompt_path if shot == 0 else self.one_shot_prompt_path
        with open(prompt_path, "r") as file:
            prompt_template = file.read()
            return prompt_template.replace("{text}", text)
    
# MODEL CLASSES
class Mistral_7B_Instruct_v0_3(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print(f"[+] Initializing Mistral_7B_Instruct_v0_3 model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_name = "mistralai-7B-instruct-v0.3" # ⚠️ make sure the saved file name matches the model name
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.tokenizer = MistralTokenizer.from_file(f"{self.model_path}/tokenizer.model.v3")
        self.model = Transformer.from_folder(self.model_path)
    
    def get_response(self, shot, user_input: str, log = True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], self.model, max_tokens=self.max_tokens, temperature=self.temperature, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        return self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    
class c4ai_command_a_03_2025(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print("[+] Initializing c4ai_command_a_03_2025 model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_name = "c4ai-command-a-03-2025"
        self.model_id = "CohereLabs/c4ai-command-a-03-2025"
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def get_response(self, shot, user_input: str, log = True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        gen_tokens = self.model.generate(input_ids, max_new_tokens=self.max_tokens, do_sample=True, temperature=self.temperature,)
        return self.tokenizer.decode(gen_tokens[0])

class Llama_3_1_8B_Instruct(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print("[+] Initializing Llama_3_1_8B_Instruct model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.model_name = "Llama-3.1-8B-Instruct"
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.pipeline = pipeline("text-generation", model=self.model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",)
    
    def get_response(self, shot, user_input: str, log = True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt},]
        messages = [{"role": "user", "content": prompt}]
        outputs = self.pipeline(messages, max_new_tokens=256,)
        return outputs[0]["generated_text"][-1]

class aya_expanse_8b(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print("[+] Initializing aya_expanse_8b model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_id = "CohereLabs/aya-expanse-8b"
        self.model_name = "aya-expanse-8b"
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
    
    def get_response(self, shot, user_input: str, log=True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        gen_tokens = self.model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=self.temperature,)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = gen_tokens[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

class Qwen2_5_7B_Instruct(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print("[+] Initializing Qwen2_5_7B_Instruct model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_name = "aya-expanse-8b"
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def get_response(self, shot, user_input: str, log = True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

class Llama_3_3_70B_Instruct(BaseModel):
    def __init__(self, zero_shot_prompt_path, one_shot_prompt_path, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 2048, temperature: float = 0.0):
        print("[+] Initializing Llama_3_3_70B_Instruct model...")
        super().__init__(zero_shot_prompt_path=zero_shot_prompt_path, one_shot_prompt_path=one_shot_prompt_path, input_csv_file_path=input_csv_file_path, output_csv_file_path=output_csv_file_path, max_tokens=max_tokens, temperature=temperature)
        self.model_id = "meta-llama/Llama-3.3-70B-Instruct"
        self.model_name = "Llama-3.3-70B-Instruct"
        self.model_path = Path.home().joinpath('models', self.model_name)
        self.pipeline = pipeline("text-generation", model=self.model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",)

    def get_response(self, shot, user_input: str, log = True) -> str:
        print(f"[+] Getting response for shot={shot} and user_input='{user_input}'...") if log else None
        prompt = self.get_prompt(user_input, shot)
        messages = [{"role": "user", "content": prompt}]
        outputs = self.pipeline(messages, max_new_tokens=256,)
        return outputs[0]["generated_text"][-1]
