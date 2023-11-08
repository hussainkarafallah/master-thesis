#!/usr/bin/env bash
#SBATCH -o eval_results
#SBATCH -A snic2022-22-394 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -t 0-01:00:00
cd ~/ondemand/cnn_english_ocr_v4
./eval.sh
