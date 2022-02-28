#!/usr/bin/env bash
MODEL_TYPE="mlp"
TOKENIZER_NAME="bert-base-uncased"
# DATASET_FOLDER="data/multi_label_datasets"
DATASET_FOLDER="/media/nvme4n1/project-textmlp/datasets/"
THRESHOLD=0.2
BATCH_SIZE=32
EPOCHS=100
RESULTS_FILE="results_mlp.csv"

for seed in 1 2 3 4 5 6; do
for DATASET in "dbpedia"; do
	python3 run_text_classification.py --dataset_folder "$DATASET_FOLDER" --model_type "$MODEL_TYPE" --threshold "$THRESHOLD" --tokenizer_name "$TOKENIZER_NAME" \
		--batch_size $BATCH_SIZE --learning_rate "0.1"\
		--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
done
