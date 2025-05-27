import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, pipeline
import torch
import random
import yaml
import re

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

def run_mc_evaluation(mc_list, model_name, api_key, name, context, country,question_type, device):
    client = OpenAI(api_key=api_key)

    global _llama_pipe, _llama_tokenizer, _llama_terminators

    
    if "gpt" in model_name.lower():
        template = load_template("../../prompt/mc_eval_template_gpt.txt")
    elif "llama" in model_name.lower():
        template = load_template("../../prompt/mc_eval_template_llama.txt")
    else:
        raise ValueError("Unsupported model name.")

    format_kwargs = {"name": name, "type": question_type, "model": model_name, "country":country}
    result_data = []
    if "gpt" in model_name.lower():
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

        return result_data

    # --- Llama path: batch processing ---
    # Initialize pipeline & tokenizer once
    if _llama_pipe is None:
        _llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llama_pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=_llama_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device
        )
        _llama_terminators = [
            _llama_tokenizer.eos_token_id,
            _llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    # Prepare all prompts up front
    all_prompts = []
    for row in mc_list:
        prompt_text = template.format(
            character=name, profile=context,
            Question=row['Question'],
            answer1=row['one'], answer2=row['two'],
            answer3=row['three'], answer4=row['four'], answer5=row['five']
        )
        all_prompts.append(f"System: I want you to act like {name}\nUser: {prompt_text}")

    # Batch size for GPU throughput (tune for your 12GB / 48GB)
    BATCH_SIZE = 8

    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch_prompts = all_prompts[i : i + BATCH_SIZE]
        batch_rows    = mc_list[i : i + BATCH_SIZE]

        outputs = _llama_pipe(
            batch_prompts,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.2,
            eos_token_id=_llama_terminators,
        )

        # 'outputs' is a list of dicts, one per prompt
        for row, out in zip(batch_rows, outputs):
            response = out[0]["generated_text"].strip()
            model_answer = response.split("\n")[-1]
            number = extract_model_answer_number(model_answer)

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

    return result_data

if __name__ == "__main__":
    cfg = load_config("../../config.yaml")

    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--question_type", type=str, default="cross")  # cross, fact, cultural, temporal
    parser.add_argument("--model_name", type=str, help="[gpt-4o, meta-llama/Llama-3.1-8B-Instruct]")
    parser.add_argument("--context_types", type=list, default=["birth", "Nationality", "Summary"])
    parser.add_argument("--meta_char_dir", type=str, default="../../data/source_data/fortesting_meta_character_2.json")
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
    output_dir = f"../../data/prediction_data/{args.model_name}/{str(args.context_types)}"
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
                            name=char_name, context=char_context, question_type=args.question_type, country=country, device=device)

            result_dic[country][character] = mc_return_list
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=2)