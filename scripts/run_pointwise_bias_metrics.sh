RANK_TYPE="pointwise"
TASK="resume_screening"
REFERENCE_GROUP="W_M"
RESULTS_DIR="hiring_pointwise_results"
OUTPUT_DIR="results/resume_screening"
TP_LABEL="high_chance"

python compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP

python compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP --true_positive_label $TP_LABEL


TASK="essay_grading"
REFERENCE_GROUP="ENS"
RESULTS_DIR="essay_pointwise_results"
OUTPUT_DIR="results/essay_grading"
TP_LABEL="Good"

python compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP

python compute_bias.py --rank_type $RANK_TYPE --task $TASK --results_dir $RESULTS_DIR --output_dir $OUTPUT_DIR \
    --reference_group $REFERENCE_GROUP --true_positive_label $TP_LABEL --load_rating