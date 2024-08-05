import os
import argparse
import pandas as pd

tasks = ["resume_screening", "essay_grading"]
reference_group = {"resume_screening": "W_M", "essay_grading": "ENS"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Reference group",
    )
    return parser.parse_args()


def compute_selection_rate_gap(df, reference_group):
    selection_rate_df = df.copy()
    groups = [g for g in selection_rate_df.group.unique() if g != reference_group]
    selection_rate_gap_df = selection_rate_df.pivot(index=["subtask", "model", "top_k_rank"], columns="group", values="selection_rate").reset_index()
    for group in groups:
        selection_rate_gap_df[group] = (selection_rate_gap_df[group] - selection_rate_gap_df[reference_group])

    selection_rate_gap_df = selection_rate_gap_df.drop(columns=[reference_group])
    selection_rate_gap_df = selection_rate_gap_df.melt(["subtask", "model", "top_k_rank"], var_name="group", value_name="selection_rate_diff").reset_index(drop=True)
    selection_rate_gap_df["selection_rate_gap"] = selection_rate_gap_df["selection_rate_diff"].apply(abs)
    return selection_rate_gap_df

def main():
    args = parse_args()

    for rank_type in ["pointwise", "pairwise"]:
        for task in tasks:
            selection_rate_df = pd.read_csv(f"{args.results_dir}/{task}/{rank_type}_top_k_ranking.csv")
            selection_rate_gap_df = compute_selection_rate_gap(selection_rate_df, reference_group[task])
            selection_rate_gap_df.to_csv(f"{args.results_dir}/{task}/{rank_type}_DP_gap.csv", index=False)

            selection_rate_df = pd.read_csv(f"{args.results_dir}/{task}/{rank_type}_top_k_ranking_TP.csv")
            selection_rate_gap_df = compute_selection_rate_gap(selection_rate_df, reference_group[task])
            selection_rate_gap_df.to_csv(f"{args.results_dir}/{task}/{rank_type}_EO_gap.csv", index=False)


if __name__ == "__main__":
    main()