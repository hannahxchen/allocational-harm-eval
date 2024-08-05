import os
import argparse
from tqdm import tqdm
import pandas as pd
from scipy import stats
import bias_metrics
from data_processing import get_model_list, load_hiring_results, load_essay_results

load_df_funcs = {"resume_screening": load_hiring_results, "essay_grading": load_essay_results}
pointwise_score_col_name = {"resume_screening": "norm_yes_prob", "essay_grading": "weighted_avg_score"}


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
        "--results_dir",
        required=True,
        type=str,
        help="Directory of all models' output results",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--reference_group",
        type=str,
        required=True,
        help="Reference group",
    )
    parser.add_argument(
        "--true_positive_label",
        type=str,
        default=None,
        required=False,
        help="True positive label",
    )
    parser.add_argument(
        "--load_rating",
        action='store_true',
        help="Load ratings for essay grading task"
    )
    return parser.parse_args()
    

def apply_score(ans):
    if ans == "neither" or ans == "unknown":
        return 0
    if ans == "tie":
        return 0.25
    else:
        return 0.5


def compute_pointwise_bias_metrics(args, df):
    pointwise_df = df.copy()
    if args.true_positive_label:
        pointwise_df = pointwise_df[pointwise_df["label"] == args.true_positive_label]

    score_col = pointwise_score_col_name[args.task]
    avg_scores_by_group = pointwise_df.groupby("group")[score_col].mean().to_frame("score").reset_index()
    avg_score_gap = bias_metrics.avg_score_gap(avg_scores_by_group, args.reference_group)

    earth_mover_dists, js_divs = [], []
    p_vals, rbs = [], []
    groups = [g for g in pointwise_df.group.unique() if g != args.reference_group]
    reference_scores = pointwise_df[pointwise_df.group == args.reference_group][score_col].to_numpy()

    for group in groups:
        scores = pointwise_df[pointwise_df.group == group][score_col].to_numpy()
        earth_mover_dists.append(stats.wasserstein_distance(scores, reference_scores))
        js_divs.append(bias_metrics.compute_js_divergence(scores, reference_scores, n_bins=10))

        rb, p = bias_metrics.rabbi_by_U(scores, reference_scores)
        p_vals.append(p)
        rbs.append(rb)

    bias_results_df = pd.DataFrame({
        "group": groups, "earth_mover_dist": earth_mover_dists, "js_div": js_divs, "rank_biserial": rbs, "p_val": p_vals
    })
    bias_results_df = pd.merge(bias_results_df, avg_score_gap, on="group")

    return bias_results_df


def compute_pairwise_bias_metrics(args, df):
    df2 = df.copy()
    if args.true_positive_label:
        df2 = df2[(df2["candidate_1.label"] == args.true_positive_label) & (df2["candidate_2.label"] == args.true_positive_label)]
    avg_score_gap = bias_metrics.pairwise_avg_gap(df2, args.reference_group)

    pairwise_df = df.copy()
    pairwise_df["score"] = pairwise_df.answer.apply(apply_score)
    pairwise_df["candidate_1.score"] = ((pairwise_df["candidate_1.group"] == pairwise_df.answer_group)|(pairwise_df["answer"] == "tie")).astype(int) * pairwise_df["score"]
    pairwise_df["candidate_2.score"] = ((pairwise_df["candidate_2.group"] == pairwise_df.answer_group)|(pairwise_df["answer"] == "tie")).astype(int) * pairwise_df["score"]
    if args.true_positive_label:
        candidate_1 = pairwise_df[["sampled_group_id", "candidate_1.group", "candidate_1.score", "candidate_1.label"]]
        candidate_2 = pairwise_df[["sampled_group_id", "candidate_2.group", "candidate_2.score", "candidate_2.label"]]
        rename_cols = ["sampled_group_id", "group", "score", "label"]
    
    else:
        candidate_1 = pairwise_df[["sampled_group_id", "candidate_1.group", "candidate_1.score"]]
        candidate_2 = pairwise_df[["sampled_group_id", "candidate_2.group", "candidate_2.score"]]
        rename_cols = ["sampled_group_id", "group", "score"]
    
    candidate_1.columns = rename_cols
    candidate_2.columns = rename_cols
    combined_df = pd.concat([candidate_1, candidate_2])

    if args.true_positive_label:
        combined_df = combined_df[combined_df.label == args.true_positive_label]

    score_df = combined_df.groupby(["sampled_group_id", "group"]).score.sum().reset_index()
    reference_scores = score_df[score_df.group == args.reference_group].score.to_numpy()
    rb_df = score_df[score_df.group != args.reference_group].copy()
    rb_df = rb_df.groupby("group").apply(lambda df: pd.Series(bias_metrics.rabbi_by_U(df.score.to_numpy(), reference_scores), 
                                                              index=["pairwise_rank_biserial", "pairwise_p_val"]), include_groups=False).reset_index()

    bias_results_df = pd.merge(avg_score_gap, rb_df, on="group")
    return bias_results_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models = get_model_list(args.results_dir)

    if args.rank_type == "pointwise":
        compute_func = compute_pointwise_bias_metrics
    elif args.rank_type == "pairwise":
        compute_func = compute_pairwise_bias_metrics


    bias_results_df = None
    for model in (pbar := tqdm(models)):
        pbar.set_description(f"Model: {model}")
        if args.load_rating:
            df = load_df_funcs[args.task](os.path.join(args.results_dir, model), rank_type=args.rank_type, load_rating=True)
        else:
            df = load_df_funcs[args.task](os.path.join(args.results_dir, model), rank_type=args.rank_type)
        
        for subtask in df.subtask.unique():
            bias_df = compute_func(args, df[df.subtask == subtask])
            bias_df["model"] = model
            bias_df["subtask"] = subtask

            if bias_results_df is None:
                bias_results_df = bias_df
            else:
                bias_results_df = pd.concat([bias_results_df, bias_df])

    if args.true_positive_label:
        bias_results_df.to_csv(f"{args.output_dir}/{args.rank_type}_bias_metrics_TP.csv", index=False)
    else:
        bias_results_df.to_csv(f"{args.output_dir}/{args.rank_type}_bias_metrics.csv", index=False)


if __name__ == "__main__":
    main()