RANK_TYPE="pairwise"
TASK="resume_screening"
REFERENCE_GROUP="W_M"
METRIC="DP"
RESULTS_DIR="hiring_pairwise_results"
OUTPUT_DIR="results/resume_screening"
TP_LABEL="high_chance"

python eval/compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP

python eval/compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP --true_positive_label $TP_LABEL


TASK="essay_grading"
REFERENCE_GROUP="ENS"
METRIC="DP"
RESULTS_DIR="essay_pairwise_results"
OUTPUT_DIR="results/essay_grading"
TP_LABEL="Good"

python eval/compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP

RESULTS_DIR="essay_pairwise_with_rating"
python eval/compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP --true_positive_label $TP_LABEL --load_rating