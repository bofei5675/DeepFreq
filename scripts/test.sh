#!/usr/bin/env bash
cd ..
python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_50_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_50_bias_no/fr/epoch_300.pth \
	--overwrite