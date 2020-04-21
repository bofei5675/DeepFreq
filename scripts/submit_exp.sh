#!/usr/bin/env bash
bias="no"
for snr in 1 5 10 20 30 40 50
do
    sbatch --export=bias=$bias,snr=$snr experiment_df.s
done
