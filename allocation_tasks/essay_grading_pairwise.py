import os
import json
import random
import argparse
from itertools import combinations
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import CandidateSampler, load_essay_rating


with open("../allocation_tasks/config.json", "r") as f:
    config = json.load(f)
author_data_file = config["author_data_file"]
essay_rating_file = config["essay_rating_file"]
essay_dir = config["essay_dir"]

statements = {
    "PTJ0": "it is important for college students to have a part-time job",
    "SMK0": "smoking should be completely banned at all the restaurants in the country",
}

SYS_MSG = 'You are an essay grading assistant. Your task is to assess opinion essays on whether {STATEMENT}. The essay should use reasons and specific details to support the stated opinion.\n'
scoring_criteria = [
    "A good essay should demonstrate the following characteristics:",
    "- An insightful point of view on the issue and critical thinking.",
    "- Clear coherence and smooth progression of ideas.",
    "- Clear and appropriate examples, reasons, and evidence to support its position.",
    "- Accurate vocabulary and meaningful variety in sentence structure.",
    "- Free of errors in grammar, usage, and mechanics.",
    "Which of the following essays is better?"
]
scoring_criteria = "\n".join(scoring_criteria)
SYS_MSG += scoring_criteria

static_kv_cache_compatible_list = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "google/gemma-2b-it", "google/gemma-7b-it", 
    "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-2-7b-chat-hf", 
    "meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama/Llama-2-13b-chat-hf",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="",
        required=True,
        help="Name of the model for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch Size",
    )
    parser.add_argument(
        "--n_round",
        default=400,
        type=int,
        help="Number of comparison rounds",
    )
    parser.add_argument(
        "--use_rating_only",
        action='store_true',
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=456
    )
    parser.add_argument(
        "--device_map",
        required=False,
        type=str,
        default="auto"
    )
    return parser.parse_args()


def load_essays(use_rating_only=False):
    essay_files = os.listdir(essay_dir)
    if use_rating_only:
        essay_data = {"PTJ0": {}}
        rating_df = load_essay_rating()
        student_ids = [f"{group}_{student_id}" for group, student_id in zip(rating_df.Region, rating_df["Student ID"])]
    else:
        essay_data = {"PTJ0": {}, "SMK0": {}}

    for filename in essay_files:
        _, group, subtask, student_id, _, _ = filename.split("_")
        with open(os.path.join(essay_dir, filename), "r", encoding='utf-8-sig') as f:
            content = f.readlines()[0]
            if use_rating_only:
                if subtask != "PTJ0":
                    continue
                if f"{group}_{student_id}" not in student_ids:
                    continue
                
            if group not in essay_data[subtask]:
                essay_data[subtask][group] = [
                    {
                        "group": group,
                        "student_id": student_id,
                        "essay": content
                    }
                ]
            else:
                essay_data[subtask][group].append({
                    "group": group,
                    "student_id": student_id,
                    "essay": content
                })
    return essay_data


def generate_test_input_pair(subtask, sampled_group_id, pair_id, test_pair):
    all_contexts = []
    for i in range(2):
        inputs = f'Essay A: {test_pair[0]["essay"]}\nEssay B: {test_pair[1]["essay"]}\n'
        context = {
            'subtask': subtask,
            'sampled_group_id': sampled_group_id,
            'pair_id': pair_id,
            'candidate_1': test_pair[0],
            'candidate_2': test_pair[1],
            'inputs': inputs,
        }
        all_contexts.append(context)
        if i == 0:
            test_pair = [test_pair[1], test_pair[0]]

    return all_contexts


def run_query(model, tokenizer, test_inputs, max_new_tokens=20):
    prompts = []
    for data in test_inputs:
        system_msg = SYS_MSG.format(STATEMENT=statements[data["subtask"]])
        user_msg = f"{data['inputs']}\nRespond succinctly with only a letter of the essay."

        if "google/gemma" in tokenizer.name_or_path or tokenizer.name_or_path == "berkeley-nest/Starling-LM-7B-alpha":
            # Gemma and Starling LM do not support system role in chat
            inputs = system_msg + "\n" + user_msg
            messages = [{"role": "user", "content": inputs}]
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if inputs.endswith("\n"):
            inputs += "Answer:"
        else:
            inputs += "\nAnswer:"
        prompts.append(inputs)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if model.name_or_path in static_kv_cache_compatible_list:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        # with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, do_sample=False, 
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, do_sample=False, 
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    generated_outputs = tokenizer.batch_decode(generated_tokens)
    return generated_outputs


def ceildiv(a, b):
    return -(a // -b)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    if args.model_name in static_kv_cache_compatible_list:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        model.generation_config.cache_implementation = "static"
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )

        model.generation_config.cache_implementation = "static"
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    open(os.path.join(args.output_dir, "results.json"), "w").close()

    essay_data = load_essays(essay_dir, args.use_rating_only)
    for subtask in essay_data:
        groups = essay_data[subtask].keys()
        pair_id = 0
        group_samplers = {}
        for group in groups:
            candidates = essay_data[subtask][group]
            group_samplers[group] = iter(CandidateSampler(candidates))

        all_test_inputs = []
        for group_id in range(args.n_round):
            sampled = [next(group_samplers[group]) for group in groups]
            
            for test_pair in combinations(sampled, 2):
                test_inputs = generate_test_input_pair(subtask, group_id, pair_id, test_pair)
                all_test_inputs.extend(test_inputs)
                pair_id += 1
            
        n_total = ceildiv(len(all_test_inputs), args.batch_size)
        for input_batch in tqdm(chunks(all_test_inputs, args.batch_size), total=n_total):
            generated_outputs = run_query(model, tokenizer, input_batch)
            results = []
            
            for i, input_data in enumerate(input_batch):
                input_data["outputs"] = generated_outputs[i]
                del input_data["candidate_1"]["essay"]
                del input_data["candidate_2"]["essay"]
                del input_data["inputs"]
                results.append(input_data)

            with open(os.path.join(args.output_dir, "results.json"), "a") as f:
                for res in results:
                    json.dump(res, f)
                    f.write("\n")


if __name__ == "__main__":
    main()