#!/usr/bin/env bash
cd ..
python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr_range_1_50/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_range_1_50_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_range_1_50_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr40/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_40_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_40_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr30/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_30_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_30_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr20/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_20_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_20_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr10/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_10_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_10_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr5/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_5_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_5_bias_no/fr/epoch_300.pth \
	--overwrite


python evaluate_bias.py \
	--data_dir test_dataset/ \
  	--output_dir results/snr1/ \
  	--fr_path1 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_1_bias_yes/fr/epoch_300.pth \
  	--fr_path2 /scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_1_bias_no/fr/epoch_300.pth \
	--overwrite