OUTPUT_DIR="hiring_pointwise_results"
MODEL_LIST="data/model_list.txt"
BATCH_SIZE=24

cat $MODEL_LIST | while read MODEL_NAME
do
    echo "Model name: $MODEL_NAME"
    model=($(echo $MODEL_NAME | tr "/" " "))
    OUTPUT_PATH="$OUTPUT_DIR/${model[1]}"
    echo "Output path: $OUTPUT_PATH"

    python resume_screening_pointwise.py --model_name $MODEL_NAME --output_dir $OUTPUT_PATH --batch_size $BATCH_SIZE
    
done