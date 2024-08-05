OUTPUT_DIR="essay_scoring_pairwise_with_rating"
MODEL_LIST="data/model_list.txt"
BATCH_SIZE=8

cat $MODEL_LIST | while read MODEL_NAME
do
    echo "Model name: $MODEL_NAME"
    model=($(echo $MODEL_NAME | tr "/" " "))
    OUTPUT_PATH="$OUTPUT_DIR/${model[1]}"
    echo "Output path: $OUTPUT_PATH"

    python essay_grading_pairwise.py --model_name $MODEL_NAME --output_dir $OUTPUT_PATH --batch_size $BATCH_SIZE --n_round 20 --use_rating_only
    
done