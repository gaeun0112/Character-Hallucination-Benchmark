import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from transformers import logging as transformers_logging 
import torch
import random
import yaml
import re
import time       
import warnings  
import logging

from huggingface_hub import login as hf_login
from huggingface_hub.utils import disable_progress_bars
from transformers.utils import logging as hf_logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 
disable_progress_bars()   

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

hf_logging.disable_progress_bar()   


_llama_pipe = None
_llama_tokenizer = None
_llama_terminators = None

def load_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def open_json(file_dir):
    with open(file_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def extract_model_answer_number(model_answer):
    match = re.search(r'\b([1-5])\b', model_answer)
    if match:
        return int(match.group(1))
    return None

def run_mc_evaluation(mc_list, model_name, api_key, name, context, country,question_type, device, hf_token):
    client = OpenAI(api_key=api_key)

    global _llama_pipe, _llama_tokenizer, _llama_terminators

    
    if "gpt" in model_name.lower():
        template = load_template("../../prompt/mc_eval_template_gpt.txt")
    elif ("llama" in model_name.lower()) or ("mistral" in model_name.lower()) or ("exaone" in model_name.lower()):
        template = load_template("../../prompt/mc_eval_template_llama.txt")
    else:
        raise ValueError("Unsupported model name.")

    format_kwargs = {"name": name, "type": question_type, "model": model_name, "country":country}
    result_data = []
    if "gpt" in model_name.lower():
        logger.info("Using GPT completion path")
        for row in mc_list:
            prompt = template.format(
                character=name, profile=context,
                Question=row['Question'],
                answer1=row['one'], answer2=row['two'],
                answer3=row['three'], answer4=row['four'], answer5=row['five']
            )
            messages = [
                {"role": "system", "content": f"I want you to act like {name}"},
                {"role": "user",   "content": prompt},
            ]
            outputs = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                n=1,
                top_p=0.95,
            )
            response = outputs.choices[0].message.content.strip()
            model_answer = response.split("\n")[-1]
            number = model_answer  # keep as string to match original
            
            result_data.append({
                "Question": row['Question'],
                "True Label": row['True Label'],
                "one": row['one'],
                "two": row['two'],
                "three": row['three'],
                "four": row['four'],
                "five": row['five'],
                "model_result": response,
                "model_answer": model_answer,
                "model_answer_number": number
            })
        logger.info("Completed GPT evaluation")
        return result_data

    # --- Llama path: batch processing ---
    # Initialize pipeline & tokenizer once
    logger.info("Using Llama batch generation path")
    if _llama_pipe is None:
        _llama_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token,    
            trust_remote_code=True    
        )
        _llama_tokenizer.pad_token = _llama_tokenizer.eos_token
        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
        _llama_model = _llama_model.to(device)

    # Prepare prompts
    prompts = []
    for row in mc_list:
        prompt_text = template.format(
            character=name, profile=context,
            Question=row['Question'],
            answer1=row['one'], answer2=row['two'],
            answer3=row['three'], answer4=row['four'], answer5=row['five']
        )
        prompts.append(prompt_text)

    # Batch inference
    BATCH_SIZE = 128  # adjust based on GPU memory
    result_data = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i: i + BATCH_SIZE]
        # 4) Batch tokenization
        inputs = _llama_tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # 5) Direct generate with optimized settings
        outputs = _llama_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            pad_token_id=_llama_tokenizer.pad_token_id,
            eos_token_id=_llama_tokenizer.eos_token_id,
            use_cache=True
        )

        # Decode and extract answer
        texts = _llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row, text in zip(mc_list[i: i + BATCH_SIZE], texts):
            # assume last line contains answer number
            model_answer = text.split("\n")[-1].strip()
            number = extract_model_answer_number(model_answer)
            result_data.append({
                "Question": row['Question'],
                "True Label": row['True Label'],
                "one": row['one'],
                "two": row['two'],
                "three": row['three'],
                "four": row['four'],
                "five": row['five'],
                "model_result": text,
                "model_answer": model_answer,
                "model_answer_number": number
            })
    logger.info("Completed Llama batch evaluation")
    return result_data

if __name__ == "__main__":
    start_time = time.time()


    cfg = load_config("../../config.yaml")
    hf_token = cfg.get('huggingface_token')  # config.yaml에 hf 토큰 키 저장 가정
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("Warning: No Hugging Face token found in config. 모델 다운로드 시 인증이 필요할 수 있습니다.")

    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--question_type", type=str, default="cross")  # cross, fact, cultural, temporal
    parser.add_argument("--model_name", type=str, help="[gpt-4o, meta-llama/Llama-3.1-8B-Instruct]")
    parser.add_argument(
            "--context_types",
            nargs="+",              # 1개 이상
            default=["birth", "Nationality", "Summary"],
            help="contexts to include, e.g. --context_types birth nationality"
        )
    parser.add_argument("--meta_char_dir", type=str, default="../../data/source_data/meta_character_2.json")
    parser.add_argument('--device_index', type=int, default=0)

    args = parser.parse_args()

    character_info = open_json(file_dir=args.meta_char_dir)
    data = open_json(file_dir=f"../../data/test_data/{args.question_type}_mc.json")

    result_dic = {}
    if args.device_index >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_index}")
    else:
        device = torch.device("cpu")

    filename = f"{args.question_type}_evaluation_result.json"
    folder_name = args.model_name
    if "gpt" not in args.model_name:
        folder_name = args.model_name.split("/")[1]
    output_dir = f"../../data/prediction_data/{folder_name}/{str(args.context_types)}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    for country in tqdm(character_info):
        result_dic[country] = {}
        for character in character_info[country]:
            if args.question_type in ["cross", "fact"]:
                if character not in data[country]:
                    continue
                mc_list_data = data[country][character]
            elif args.question_type == "cultural":
                mc_list_data = data[country]
            elif args.question_type == "temporal":
                mc_list_data = data
            
            # 캐릭터 name, context, profile 정의 
            char_name = character
            char_profile = character_info[country][character]['profile']
            char_context = {}
            for context_label in args.context_types:
                char_context[context_label] = character_info[country][character]['context'][context_label]
            if len(args.context_types)==0:
                char_context = char_profile


            mc_return_list = run_mc_evaluation(mc_list=mc_list_data, model_name=args.model_name, api_key=cfg["openai_key"],
                            name=char_name, context=char_context, question_type=args.question_type, country=country, device=device, hf_token=hf_token)

            result_dic[country][character] = mc_return_list
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=2)

    end_time = time.time()      
    total_sec = end_time - start_time 
    print(f"Total execution time: {total_sec:.2f} seconds")