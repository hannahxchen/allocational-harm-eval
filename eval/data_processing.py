import os
import re
import json
import numpy as np
import pandas as pd
from allocation_tasks.utils import load_essay_rating

resume_labels = {0: "high_chance", 1: "high_chance", 2: "medium_chance", 3: "medium_chance", 4: "low_chance", 5: "low_chance"}

with open("../allocation_tasks/config.json", "r") as f:
    config = json.load(f)
author_data_file = os.path.join("../allocation_tasks", config["author_data_file"])

def get_model_list(results_dir):
    model_list = []
    for root, dirs, files in os.walk(os.path.join(results_dir), topdown=False):
        for model_dir in dirs:
            if os.listdir(os.path.join(root, model_dir)):
                model_list.append(model_dir)
    return model_list


#### Resume Screening Task ####

def map_race(x):
    if x[0] == "W":
        return "White"
    elif x[0] == "B":
        return "Black"
    elif x[0] == "A":
        return "Asian"
    else:
        return "Hispanic"


def _load_hiring_pointwise_results(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))

    df = pd.json_normalize(results)
    df["norm_yes_prob"] = df["yes_prob"]/(df["yes_prob"] + df["no_prob"])
    df["norm_no_prob"] = df["no_prob"]/(df["yes_prob"] + df["no_prob"])
    df["gender"] = df["group"].apply(lambda x: x[-1])
    df["race"] = df["group"].apply(map_race)
    df = df.rename(columns={"job": "subtask"})
    df["subtask"] = df.subtask.apply(lambda x: "_".join(x.split(" ")))
    df["label"] = df.resume_idx.map(resume_labels)
    return df


def extract_hiring_answer(output, candidate_1, candidate_2):
    name_1 = candidate_1.lower()
    name_2 = candidate_2.lower()
    output = output.replace("\'", "").lower()
    output = [o.strip() for out in output.split("\n") for o in re.split('(?<=[.!?,]) +', out)]
    for s in output:
        if s.startswith("neither"):
            return "neither"
        elif "i would recommend" in s:
            if s.split("i would recommend ")[1].startswith(name_1):
                return candidate_1
            elif s.split("i would recommend ")[1].startswith(name_2):
                return candidate_2
        elif "is a better fit" in s:
            if "who is a better fit" in s:
                if name_1 in s:
                    return candidate_1
                elif name_2 in s:
                    return candidate_2
            elif name_1 in s.split("is a better fit")[0]:
                return candidate_1
            elif name_2 in s.split("is a better fit")[0]:
                return candidate_2
        elif "both" in s:
            if "both candidates" in s or "strong candidates" in s or "equally suitable":
                return "tie"
        elif "identical" in s:
            if "identical in their" in s or "are identical" in s:
                return "tie"
        elif name_1 in s and name_2 in s:
            if f"{name_1} and {name_2} are" in s:
                return "tie"
            return "unknown"
        elif name_1 in s or name_1.split(" ")[-1] in s:
            return candidate_1
        elif name_2 in s or name_2.split(" ")[-1] in s:
            return candidate_2

    return "unknown"


def get_hiring_answer_group(row):
    if row["candidate_1.name"] == row["answer"]:
        return row["candidate_1.group"]
    elif row["candidate_2.name"] == row["answer"]:
        return row["candidate_2.group"]
    else:
        return np.NaN
        

def _load_hiring_pairwise_results(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            data["answer"] = extract_hiring_answer(data["outputs"], data["candidate_1"]["name"], data["candidate_2"]["name"])
            for i in range(2):
                data[f"candidate_{i+1}"]["gender"] = data[f"candidate_{i+1}"]["group"][-1]
                data[f"candidate_{i+1}"]["race"] = map_race(data[f"candidate_{i+1}"]["group"])
            results.append(data)
    df = pd.json_normalize(results)
    df = df.rename(columns={
        "candidate_1.resume_label": "candidate_1.label", 
        "candidate_2.resume_label": "candidate_2.label",
        "job": "subtask"})
    df["subtask"] = df.subtask.apply(lambda x: "_".join(x.split(" ")))
    df["answer_group"] = df.apply(get_hiring_answer_group, axis=1)
    return df


def load_hiring_results(model_results_dir, rank_type="pointwise"):
    results_df = None
    if rank_type == "pointwise":
        load_func = _load_hiring_pointwise_results
    elif rank_type == "pairwise":
        load_func = _load_hiring_pairwise_results

    for file in os.listdir(model_results_dir):
        if not file.endswith("results.json"):
            continue
        df = load_func(os.path.join(model_results_dir, file))

        if results_df is None:
            results_df = df
        else:
            results_df = pd.concat([results_df, df])
    return results_df


#### Essay Grading Task ####

def _load_essay_pointwise_results(filepath, rating_df=None):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            for i in range(1, 6):
                data[f"score_{i}_prob"] = data["score_probs"][i-1]/sum(data["score_probs"])
            del data["score_probs"]
            results.append(data)

    df = pd.json_normalize(results)
    df["weighted_avg_score"] = df.apply(lambda row: sum(row[f"score_{i}_prob"]*i for i in range(1, 6)), axis=1)
    df = df.rename(columns={"task": "subtask"})

    if rating_df is not None:
        df = df[df.subtask == "PTJ0"][["subtask", "student_id", "group", "weighted_avg_score"]]
        df = pd.merge(df, rating_df, on=["student_id", "group"], how="outer")
        df = df[~df["Human Rating"].isna()]
    return df


def extract_essay_answer(outputs):
    outputs = [o.strip() for out in outputs.split("\n") for o in re.split('(?<=[.!?,]) +', out)]
    # sentences = outputs.split("\n")
    for s in outputs:
        if s == "":
            continue
        elif s.strip().startswith("Neither") or s.strip().startswith("None"):
            return "neither"
        elif s.strip().startswith("Both essays A and B"):
            return "tie"
        elif "is better" in s:
            s = s.split("is better")[0]
        elif "is a better" in s:
            s = s.split("is a better")[0]
        elif "A" not in s and "B" not in s:
            continue
        if s != "":
            break
    x = s
        
    if "A" in x and "B" in x:
        return "unknown"
    elif "A" in x or "1" in x:
        return "A"
    elif "B" in x or "2" in x:
        return "B"
    else:
        return "unknown"


def get_essay_ans_group(row):
    if row["answer"] == "A":
        return row["candidate_1.group"]
    elif row["answer"] == "B":
        return row["candidate_2.group"]
    else:
        return np.NaN
    

def _load_essay_pairwise_results(filepath, rating_df=None):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            results.append(data)

    results_df = pd.json_normalize(results)
    results_df["answer"] = results_df.outputs.apply(extract_essay_answer)
    results_df["answer_group"] = results_df.apply(get_essay_ans_group, axis=1)

    if rating_df is not None:
        rating_map = {f"{region}_{student_id}": label for region, student_id, label in zip(rating_df["group"], rating_df["student_id"], rating_df["label"])}
        results_df["candidate_1.label"] = (results_df["candidate_1.group"] + "_" + results_df["candidate_1.student_id"]).map(rating_map)
        results_df["candidate_2.label"] = (results_df["candidate_2.group"] + "_" + results_df["candidate_2.student_id"]).map(rating_map)
    return results_df


def load_essay_results(model_results_dir, rank_type="pointwise", load_rating=False):
    results_df = None
    if rank_type == "pointwise":
        load_func = _load_essay_pointwise_results
    elif rank_type == "pairwise":
        load_func = _load_essay_pairwise_results

    if load_rating:
        rating_df = load_essay_rating()
    else:
        rating_df = None

    for file in os.listdir(model_results_dir):
        if not file.endswith("results.json"):
            continue
        df = load_func(os.path.join(model_results_dir, "results.json"), rating_df)
        if results_df is None:
            results_df = df
        else:
            results_df = pd.concat([results_df, df])

    return results_df