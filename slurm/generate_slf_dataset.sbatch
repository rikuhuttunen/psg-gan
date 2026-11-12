#!/bin/bash
#SBATCH --job-name generate_slf_dataset
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 1200
#SBATCH --mem=50G
#SBATCH --partition gpu
#SBATCH --output slurm_outputs/generate_slf_dataset_%j.txt
#SBATCH --error slurm_outputs/generate_slf_dataset_%j.txt
#SBATCH --gres=gpu:1

# Log GPU details
nvidia-smi

# Run generation
/userapps/rikuhut/.local/bin/hatch run python src/psg_gan/stylegan3/generate_slf_dataset.py -n ${NSAMPLES:-1000} -b ${BATCH_SIZE:-8} \
    --ds-base-dir ${DS_BASE_DIR} \
    --ds-name ${DS_NAME:-$(date +%F)_stylegan3} --series-name ${SERIES_NAME:-stylegan3} \
    --gan-ckpt-dir ${GAN_CKPT_DIR} \
    --scorer-ckpt-dir ${SCORER_CKPT_DIR:-''} \
    --scorer-name ${SCORER_NAME:-'utime'}
