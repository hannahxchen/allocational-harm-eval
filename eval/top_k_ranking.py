import os
import random
import argparse
from tqdm import tqdm, trange
from collections import Counter
import pandas as pd
from data_processing import get_model_list, load_hiring_results, load_essay_results
from allocation_tasks.utils import CandidateSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank_type",
        required=True,
        choices=["pointwise", "pairwise"],
        help="Pointwise or Pairwise ranking"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["resume_screening", "essay_grading"],
        help="Task name"
    )
    parser.add_argument(
        "--true_positive_label",
        type=str,
        default=None,
        required=False,
        help="True positive label",
    )
    parser.add_argument(
        "--max_top_k",
        default=5,
        type=int,
        required=False,
        help="Maximum Top k for ranking"
    )
    parser.add_argument(
        "--rounds",
        default=12000,
        type=int,
        required=False,
        help="Number of pointwise ranking rounds"
    )
    parser.add_argument(
        "--results_dir",
        required=False,
        default="results",
        type=str,
        help="Directory that contains model output results",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        default=4810,
        type=int,
        required=False,
        help="Random seed",
    )
    return parser.parse_args()
    

def apply_score(ans):
    if ans in ["neither", "unknown"]:
        return 0
    if ans == "tie":
        return 0.25
    else:
        return 0.5
    

def pointwise_ranking(args, df, sample_by_group=True, pool_size=10):
    pointwise_df = df.copy()
    if args.true_positive_label:
        groups = df[df.label == args.true_positive_label].group.unique()
        candidate_cols = ["score", "label", "group"]
    else:
        groups = df.group.unique()
        candidate_cols = ["score", "group"]

    top_rank_counts = {g: [0] * args.max_top_k for g in groups}
    total_counts = Counter({g: 0 for g in groups})

    if sample_by_group:
        pool_size = len(groups)
        group_samplers = {}
        if sample_by_group:
            for group in groups:
                candidates = pointwise_df[(pointwise_df.group == group)][candidate_cols].apply(lambda row: list(row), axis=1).tolist()
                group_samplers[group] = iter(CandidateSampler(candidates))
    else:
        candidates = pointwise_df[candidate_cols].apply(lambda row: list(row), axis=1).tolist()
        sampler = CandidateSampler(candidates)

    for _ in trange(args.rounds, desc="Running Pointwise Ranking", leave=False):
        if sample_by_group:
            candidates = [next(group_samplers[group]) for group in group_samplers]
        else:
            candidates = [next(sampler) for _  in range(pool_size)]
        sampled_candidates = pd.DataFrame({"group": [c[-1] for c in candidates], "score": [c[0] for c in candidates]})

        if args.true_positive_label:
            sampled_candidates["label"] = [c[1] for c in candidates]
            counts = Counter(sampled_candidates[sampled_candidates.label == args.true_positive_label].value_counts("group").to_dict())
        else:
            counts = Counter(sampled_candidates.value_counts("group").to_dict())

        total_counts += counts
        sampled_candidates = sampled_candidates.sort_values("score", ascending=False).reset_index(drop=True)

        for top_k in range(args.max_top_k):
            top_rank = sampled_candidates[sampled_candidates["score"] > sampled_candidates.iloc[top_k]["score"]].reset_index()
            tie_rank = sampled_candidates[sampled_candidates["score"] == sampled_candidates.iloc[top_k]["score"]].reset_index()
            k = top_rank.shape[0]
            n_ties = tie_rank.shape[0]
            
            if args.true_positive_label:
                top_rank = top_rank[top_rank["label"] == args.true_positive_label]
                tie_rank = tie_rank[tie_rank["label"] == args.true_positive_label]

            for group in top_rank.group:
                top_rank_counts[group][top_k] += 1
            for group in tie_rank.group:
                top_rank_counts[group][top_k] += (top_k + 1 - k) / n_ties

    selection_rate_df = None
    for i in range(args.max_top_k):
        temp = pd.DataFrame({
            "group": (top_rank_counts.keys()), 
            "selection_rate": [top_rank_counts[group][i] / total_counts[group] for group in top_rank_counts]
        })
        temp["top_k_rank"] = i+1
        if selection_rate_df is None:
            selection_rate_df = temp
        else:
            selection_rate_df = pd.concat([selection_rate_df, temp])
    return selection_rate_df


def run_pointwise_top_k(args):
    ranking_df = None
    models = get_model_list(args.results_dir)
    if args.task == "resume_screening":
        score_col = "norm_yes_prob"
    elif args.task == "essay_grading":
        score_col = "weighted_avg_score"
        load_rating = False
        if args.true_positive_label:
            load_rating = True

    for model in (pbar := tqdm(models)):
        pbar.set_description(f"Model: {model}")
        if args.task == "essay_grading":
            sample_by_group = False
            results_df = load_essay_results(os.path.join(args.results_dir, model), rank_type="pointwise", load_rating=load_rating)
        else:
            sample_by_group = True
            results_df = load_hiring_results(os.path.join(args.results_dir, model), rank_type="pointwise")
            
        results_df = results_df.rename(columns={score_col: "score"})

        for subtask in (subtask_pbar := tqdm(results_df.subtask.unique(), leave=False)):
            subtask_pbar.set_description(f"Subtask: {subtask}")
            df = results_df[results_df.subtask == subtask].copy()
            temp = pointwise_ranking(args, df, sample_by_group=sample_by_group)
            temp["model"] = model
            temp["subtask"] = subtask
            if ranking_df is None:
                ranking_df = temp
            else:
                ranking_df = pd.concat([ranking_df, temp])

    return ranking_df


def pairwise_ranking(args, df):
    pairwise_df = df.copy()
    if args.true_positive_label:
        temp = pairwise_df[["sampled_group_id", "candidate_1.group", "candidate_1.label"]].drop_duplicates()
        temp = temp[temp[f"candidate_1.label"] == args.true_positive_label]
        TP_label_groups = temp.groupby("sampled_group_id")[f"candidate_1.group"].apply(list).to_dict()

    pairwise_df["score"] = pairwise_df["answer"].apply(apply_score)
    regular_ans = pairwise_df[~pairwise_df["answer_group"].isna()][["sampled_group_id", f"answer_group", "score"]]

    other_1 = pairwise_df[pairwise_df["answer_group"].isna()][["sampled_group_id", f"candidate_1.group", "score"]]
    other_2 = pairwise_df[pairwise_df["answer_group"].isna()][["sampled_group_id", f"candidate_2.group", "score"]]
    other_1 = other_1.rename(columns={"candidate_1.group": "answer_group"})
    other_2 = other_1.rename(columns={"candidate_2.group": "answer_group"})
    other_ans = pd.concat([other_1, other_2])

    score_df = pd.concat([regular_ans, other_ans]).groupby(["sampled_group_id", f"answer_group"]).score.sum().reset_index()
    score_df = score_df.rename(columns={f"answer_group": "group"})

    top_rank_by_group = {group: [0] * args.max_top_k for group in score_df.group.unique()}
    for group_id in pairwise_df.sampled_group_id.unique():
        if args.true_positive_label and group_id not in TP_label_groups:
            continue
        temp = score_df[score_df.sampled_group_id == group_id]
        temp = temp.sort_values("score", ascending=False).reset_index(drop=True)
        
        for top_k in range(args.max_top_k):
            top_rank = temp[temp.score > temp.iloc[top_k]["score"]].reset_index()
            tie_rank = temp[temp.score == temp.iloc[top_k]["score"]].reset_index()
            k = top_rank.shape[0]
            n_ties = tie_rank.shape[0]

            if args.true_positive_label:
                top_rank = top_rank[top_rank.group.isin(TP_label_groups[group_id])]
                tie_rank = tie_rank[tie_rank.group.isin(TP_label_groups[group_id])]

            for group in top_rank.group:
                top_rank_by_group[group][top_k] += 1
            for group in tie_rank.group:
                top_rank_by_group[group][top_k] += (top_k + 1 - k) / n_ties

    temp = pairwise_df.drop_duplicates(["sampled_group_id", "candidate_1.group"])
    if args.true_positive_label:
        total_counts = temp[temp["candidate_1.label"] == args.true_positive_label].groupby("candidate_1.group").size().to_dict()
    else:
        total_counts = temp.groupby("candidate_1.group").size().to_dict()

    ranking_df = None
    for i in range(args.max_top_k):
        temp = pd.DataFrame({"group": list(total_counts.keys())})
        temp["selection_rate"] = [top_rank_by_group[group][i]/total_counts[group] for group in total_counts]
        temp["top_k_rank"] = i + 1
        if ranking_df is None:
            ranking_df = temp
        else:
            ranking_df = pd.concat([ranking_df, temp])
    
    return ranking_df


def run_pairwise_top_k(args):
    models = get_model_list(args.results_dir)
    pairwise_top_k_df = None
    if args.task == "essay_grading" and args.true_positive_label:
        load_rating = True
    else:
        load_rating = False

    for model in (pbar := tqdm(models)):
        pbar.set_description(f"Model: {model}")
        if args.task == "essay_grading":
            results_df = load_essay_results(os.path.join(args.results_dir, model), rank_type="pairwise", load_rating=load_rating)
        else:
            results_df = load_hiring_results(os.path.join(args.results_dir, model), rank_type="pairwise")

        for subtask in (subtask_pbar := tqdm(results_df.subtask.unique(), leave=False)):
            subtask_pbar.set_description(f"Subtask: {subtask}")

            df = results_df[results_df.subtask == subtask]
            ranking_df = pairwise_ranking(args, df)
            ranking_df["model"] = model
            ranking_df["subtask"] = subtask

            if pairwise_top_k_df is None:
                pairwise_top_k_df = ranking_df
            else:
                pairwise_top_k_df = pd.concat([pairwise_top_k_df, ranking_df])

    return pairwise_top_k_df


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.rank_type == "pointwise":
        ranking_df = run_pointwise_top_k(args)
    elif args.rank_type == "pairwise":
        ranking_df = run_pairwise_top_k(args)

    if args.true_positive_label:
        outfile_name = f"{args.rank_type}_top_k_ranking_TP.csv"
    else:
        outfile_name = f"{args.rank_type}_top_k_ranking.csv"
    ranking_df.to_csv(os.path.join(args.output_dir, outfile_name), index=False)

if __name__ == "__main__":
    main()