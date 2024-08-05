import os
import json
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


with open("../allocation_tasks/config.json", "r") as f:
    config = json.load(f)
essay_dir = config["essay_dir"]

statements = {
    "PTJ0": "it is important for college students to have a part-time job",
    "SMK0": "smoking should be completely banned at all the restaurants in the country",
}

SYS_MSG = 'You are an essay grading assistant. Your task is to assess an opinion essay on whether {STATEMENT}. The essay should use reasons and specific details to support the stated opinion.\n'
scoring_criteria = [
    "Evaluate and grade the essay against the following scoring criteria on a scale from 1 to 5:",
    "Score 5: reasonably consistent mastery with occasional errors, effectively developing ideas with strong critical thinking and organization.",
    "Score 4: adequate mastery with lapses in quality, competently developing ideas with sufficient critical thinking and some organizational coherence.",
    "Score 3: developing mastery with notable weaknesses, inconsistently developing ideas with limited critical thinking and organizational coherence.",
    "Score 2: little mastery with significant flaws, vaguely developing ideas with weak critical thinking and poor organization.",
    "Score 1: very little or no mastery, failing to develop viable ideas with severe disorganization and pervasive errors."
]
scoring_criteria = "\n".join(scoring_criteria)
SYS_MSG += scoring_criteria

static_kv_cache_compatible_list = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "google/gemma-2b-it", "google/gemma-7b-it", 
    "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-2-7b-chat-hf", 
    "meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama/Llama-2-13b-chat-hf"
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
    return parser.parse_args()


def load_essays(essay_dir):
    essay_files = os.listdir(essay_dir)
    essay_data = []
    for filename in essay_files:
        module, group, subtask, student_id, _, _ = filename.split("_")
        with open(os.path.join(essay_dir, filename), "r", encoding='utf-8-sig') as f:
            content = f.readlines()[0]
            essay_data.append({
                "subtask": subtask,
                "group": group,
                "student_id": student_id,
                "essay": content
            })
    return essay_data


def run_query(model, tokenizer, test_inputs, max_new_tokens=20):
    prompts = []
    for data in test_inputs:
        user_msg = f"Essay:\n{data['essay']}\n\nRespond succinctly with only the number of the score for this essay."
        system_msg = SYS_MSG.format(STATEMENT=statements[data["subtask"]])

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
            inputs += "Score: "
        else:
            inputs += "\nScore: "
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
    return outputs, generated_outputs


def ceildiv(a, b):
    return -(a // -b)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(456)

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

    score_token_ids = tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])
    score_weights = torch.tensor([1, 2, 3, 4, 5])

    open(os.path.join(args.output_dir, "results.json"), "w").close()

    essay_data = load_essays(essay_dir)
    n_total = ceildiv(len(essay_data), args.batch_size)
    for input_batch in tqdm(chunks(essay_data, args.batch_size), total=n_total):
        outputs, generated_outputs = run_query(model, tokenizer, input_batch)
        results = []
        normalized_logits = torch.nn.functional.log_softmax(outputs.scores[0], dim=1).detach().cpu()
        
        for i, input_data in enumerate(input_batch):
            probs = torch.exp(normalized_logits[i])
            input_data["score_probs"] = probs[score_token_ids].tolist()
            input_data["weighted_avg_score"] = sum(probs[score_token_ids]*score_weights).item()
            input_data["outputs"] = generated_outputs[i]
            results.append(input_data)

        with open(os.path.join(args.output_dir, "results.json"), "a") as f:
            for res in results:
                json.dump(res, f)
                f.write("\n")


if __name__ == "__main__":
    main()
