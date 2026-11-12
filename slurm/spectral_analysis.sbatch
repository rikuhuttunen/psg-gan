#!/bin/bash
#SBATCH --job-name spectral-analysis
#SBATCH --ntasks 1
#SBATCH --time 1200
#SBATCH --mem=50G
#SBATCH --partition gpu
#SBATCH --output slurm_outputs/spectral_analysis_%j.txt
#SBATCH --error slurm_outputs/spectral_analysis_%j.txt
#SBATCH --gres=gpu:1

# Log GPU details
nvidia-smi

# Run generation
/userapps/rikuhut/.local/bin/hatch run python src/psg_gan/spectral_analysis.py \
    --cfg-path ${CFG_PATH} \
    --fig-savedir ${FIG_SAVEDIR} \
    --cache-path ${CACHE_PATH:-'/tmp/psggan_spectral_analysis.npz'}
    #--copy-dst ${COPY_DST} \
