RANK_TYPE="pairwise"
TASK="resume_screening"
RESULTS_DIR="hiring_pairwise_results"
OUTPUT_DIR="results/resume_screening"
MAX_TOP_K=5
TP_LABEL="high_chance"

python eval/top_k_ranking.py --rank_type $RANK_TYPE --task $TASK --max_top_k $MAX_TOP_K \
    --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR

python eval/top_k_ranking.py --rank_type $RANK_TYPE --task $TASK --max_top_k $MAX_TOP_K \
    --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR --true_positive_label $TP_LABEL

TASK="essay_grading"
RESULTS_DIR="essay_pairwise_results"
OUTPUT_DIR="results/essay_grading"
TP_LABEL="Good"

python eval/top_k_ranking.py --rank_type $RANK_TYPE --task $TASK --max_top_k $MAX_TOP_K \
    --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR

RESULTS_DIR="essay_pairwise_with_rating"
python eval/top_k_ranking.py --rank_type $RANK_TYPE --task $TASK --max_top_k $MAX_TOP_K \
    --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR --true_positive_label $TP_LABEL