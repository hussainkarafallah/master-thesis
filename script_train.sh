#!/usr/bin/env bash
#SBATCH -o train_results
#SBATCH -A snic2022-22-394 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -t 7-00:00:00

cd ~/ondemand/cnn_english_ocr_v4

apptainer exec ../my_container.sif python3 train.py --epoch 300 --save-to ~/ondemand/cnn_english_ocr_v4/checkpoints \
                 --train-data ~/ondemand/data/ocr_data/data/train_data.csv \
                 --val-data ~/ondemand/data/ocr_data/data/val_data.csv \
                 --data ~/ondemand/data/ocr_data/data \
                 --vocab ~/ondemand/data/ocr_data/data/char_dict.json \
                 --save-freq 30493 \
                 --lr 1e-4 \
                 --ngpus 2
