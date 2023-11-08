cd ~/ondemand/cnn_english_ocr_v4

apptainer exec ../my_container.sif python3 train.py --epoch 30 --save-to ~/ondemand/cnn_english_ocr_v4/checkpoints \
                 --train-data ~/ondemand/data/ocr_data/data/train_data.csv \
                 --val-data ~/ondemand/data/ocr_data/data/val_data.csv \
                 --data ~/ondemand/data/ocr_data/data \
                 --vocab ~/ondemand/data/ocr_data/data/char_dict.json \
                 --save-freq 5000 \
                 --lr 1e-3 \
		         --cpt ~/ondemand/cnn_english_ocr_v4/checkpoints/cp_26_0.pth \
                 --eval True \
                 --ngpus 2
