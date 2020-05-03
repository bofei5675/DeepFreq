#!/bin/bash
#SBATCH --job-name=ds-ga-1013-deep-freq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=bz1030@nyu.edu
#SBATCH --output=df_%j.out
#SBATCH --error=df_%j.err

module purge
source /scratch/bz1030/capstone_env/bin/activate

bash test.sh