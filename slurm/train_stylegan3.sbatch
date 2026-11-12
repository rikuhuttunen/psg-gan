#!/bin/bash
#SBATCH --job-name train-psg-gan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 3600
#SBATCH --mem=100G
#SBATCH --partition gpu
#SBATCH --output slurm_outputs/train_stylegan3_%j.txt
#SBATCH --error slurm_outputs/train_stylegan3_%j.txt
#SBATCH --gres=gpu:1

# Skip copying by setting COPY_DATASETS to other than 1
if [[ "${COPY_DATASETS:-1}" == "1" ]]; then
    # Copy data for faster access, e.g. to /scratch
    DST_DIR=${DS_DST_DIR:-/local_scratch/rikuhut/data/slf}
    mkdir -p $DST_DIR

    STAGES_SRC_DIR=${STAGES_SRC_DIR:-/research/work/rikuhut/data/slf/STAGES_crop_lights_off_on}
    echo "Copying data from $STAGES_SRC_DIR to $DST_DIR"
    rsync -r --update $STAGES_SRC_DIR $DST_DIR
else
    printf 'Skip copying datasets...'
fi

# Log GPU details
nvidia-smi

# Run training
/userapps/rikuhut/.local/bin/hatch run python src/psg_gan/train.py -p $PARAM_PATH \
    --visible-device 0 --mlflow-tracking-uri ${MLFLOW_TRACKING_URI:-http://localhost:5000} \
    --mlflow-experiment ${MLFLOW_EXPERIMENT:-'psg-gan-new'}
