import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
import torch
import random
import yaml
import re
import time
import warnings
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def run_mc_evaluation(mc_list, model_name, name, context, template, is_gpt=False,
                      client=None, llm=None, sampling_params=None,
                      batch_size=64, temperature=0.0, seed=42):
    set_seed(seed)
    result_data = []


    if is_gpt:
        for row in mc_list:
            prompt = template.format(
                character=name, profile=context,
                Question=row['Question'],
                answer1=row['one'], answer2=row['two'],
                answer3=row['three'], answer4=row['four'], answer5=row['five']
            )
            messages = [
                {"role": "system", "content": f"I want you to act like {name}"},
                {"role": "user", "content": prompt},
            ]

            # with open('./prompt_text.txt', "w", encoding="utf-8") as f:
            #     f.write("----- PROMPT -----\n")
            #     f.write(prompt + "\n\n")
            #     f.write("----- MESSAGES -----\n")
            #     # JSON 형태로 보기 좋게 저장하고 싶으면 json.dumps 사용
            #     f.write(json.dumps(messages, ensure_ascii=False, indent=2))

            outputs = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
                n=1,
                top_p=0.95,
            )
            response = outputs.choices[0].message.content.strip()
            model_answer = response.split("\n")[-1]
            result_data.append({
                "Question": row['Question'],
                "True Label": row['True Label'],
                "one": row['one'], "two": row['two'], "three": row['three'],
                "four": row['four'], "five": row['five'],
                "model_result": response,
                "model_answer": model_answer,
                "model_answer_number": extract_model_answer_number(model_answer)
            })
    else:
        prompts = []
        for row in mc_list:
            prompt_text = template.format(
                character=name, profile=context,
                Question=row['Question'],
                answer1=row['one'], answer2=row['two'],
                answer3=row['three'], answer4=row['four'], answer5=row['five']
            )
            prompts.append((prompt_text, row))

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            prompt_texts = [p[0] for p in batch]
            rows = [p[1] for p in batch]

            outputs = llm.generate(prompt_texts, sampling_params)

            for row, output in zip(rows, outputs):
                generated_text = output.outputs[0].text.strip()
                model_answer = generated_text.split("\n")[-1].strip()
                result_data.append({
                    "Question": row['Question'],
                    "True Label": row['True Label'],
                    "one": row['one'], "two": row['two'], "three": row['three'],
                    "four": row['four'], "five": row['five'],
                    "model_result": generated_text,
                    "model_answer": model_answer,
                    "model_answer_number": extract_model_answer_number(model_answer)
                })
    return result_data

if __name__ == "__main__":
    start_time = time.time()

    cfg = load_config("../../config.yaml")

    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--question_type", type=str, default="cross")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--context_types", nargs="*", default=["birth", "Nationality", "Summary"])
    parser.add_argument("--meta_char_dir", type=str, default="../../data/source_data/meta_character_2.json")
    parser.add_argument("--input_dir_path", type=str, default="../../data/test_data")
    parser.add_argument("--device_index", type=str, help="GPU device indices, comma-separated (예: 0,1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    character_info = open_json(file_dir=args.meta_char_dir)
    data = open_json(file_dir=f"{args.input_dir_path}/{args.question_type}_mc.json")

    result_dic = {}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device_indices = args.device_index.split(",")
    tensor_parallel_size = len(device_indices)

    filename = f"{args.question_type}_evaluation_result.json"
    folder_name = args.model_name.split("/")[-1] if "gpt" not in args.model_name else args.model_name
    if args.input_dir_path.split("/")[-1] != "test_data":
        folder_name = f"{folder_name}_{args.input_dir_path.split('/')[-1]}"
    output_dir = f"../../data/prediction_data/{folder_name}/{str(args.context_types)}"
    # output_dir = f"../../data/prediction_data/{folder_name}_2/{str(args.context_types)}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Model & template loading (only once)
    is_gpt = "gpt" in args.model_name.lower()
    client = OpenAI(api_key=cfg["openai_key"]) if is_gpt else None
    template_path = "../../prompt/mc_eval_template_gpt.txt" if is_gpt else "../../prompt/mc_eval_template_llama.txt"
    template = load_template(template_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=64,
        stop=[],
        seed=args.seed
    ) if not is_gpt else None
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16"
    ) if not is_gpt else None

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

            char_name = character
            char_profile = character_info[country][character]['profile']
            if args.context_types[0] == "no_context":
                char_context = char_profile
            else:
                char_context = {label: character_info[country][character]['context'][label] for label in args.context_types}
        

            mc_return_list = run_mc_evaluation(
                mc_list=mc_list_data,
                model_name=args.model_name,
                name=char_name,
                context=char_context,
                template=template,
                is_gpt=is_gpt,
                client=client,
                llm=llm,
                sampling_params=sampling_params,
                batch_size=32,
                temperature=args.temperature,
                seed=args.seed
            )

            result_dic[country][character] = mc_return_list
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
