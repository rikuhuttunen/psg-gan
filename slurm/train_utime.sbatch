#!/bin/bash
#SBATCH --job-name train-utime
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 1200
#SBATCH --mem=100G
#SBATCH --partition gpu
#SBATCH --output slurm_outputs/train_utime_%j.txt
#SBATCH --error slurm_outputs/train_utime_%j.txt
#SBATCH --gres=gpu:1

# Skip copying by setting COPY_DATASETS to e.g. "0"
if [[ "${COPY_DATASETS:-1}" == "1" ]]; then
    # Copy data for faster access, e.g. to /scratch
    DST_DIR=${DS_DST_DIR:-/local_scratch/rikuhut/data/slf}
    mkdir -p $DST_DIR

    MESA_SRC_DIR=${MESA_SRC_DIR:-/research/work/rikuhut/data/slf/MESA_crop_lights_off_on}
    echo "Copying data from $MESA_SRC_DIR to $DST_DIR"
    rsync -r --update --exclude='*_iqr*' --exclude='*_unnorm*' $MESA_SRC_DIR $DST_DIR

    STAGES_SRC_DIR=${STAGES_SRC_DIR:-/research/work/rikuhut/data/slf/STAGES_crop_lights_off_on}
    echo "Copying data from $STAGES_SRC_DIR to $DST_DIR"
    rsync -r --update $STAGES_SRC_DIR $DST_DIR

    STAR_SA_SRC_DIR=${STAR_SA_SRC_DIR:-/research/work/rikuhut/data/slf/STAR-SA_extracted}
    echo "Copying data from $STAR_SA_SRC_DIR to $DST_DIR"
    rsync -r --update $STAR_SA_SRC_DIR $DST_DIR

    STAR_HV_SRC_DIR=${STAR_HV_SRC_DIR:-/research/work/rikuhut/data/slf/STAR_HV_extracted}
    echo "Copying data from $STAR_HV_SRC_DIR to $DST_DIR"
    rsync -r --update $STAR_HV_SRC_DIR $DST_DIR

    SYNTHETIC_SRC_DIR=${SYNTHETIC_SRC_DIR:-/research/work/rikuhut/data/slf/20261006_painted-eel-654_ema_19760}
    echo "Copying data from $SYNTHETIC_SRC_DIR to $DST_DIR"
    rsync -r --update $SYNTHETIC_SRC_DIR $DST_DIR
else
    printf 'Skip copying datasets...'
fi

# Log GPU details
nvidia-smi

# Run training
# In the comments, there are all kinds of flags to debug or improve bad GPU performance
#TF_GPU_ALLOCATOR="cuda_malloc_async"
#XLA_PYTHON_CLIENT_PREALLOCATE=false
#XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
#XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" \
#XLA_FLAGS="--xla_gpu_autotune_level=0" \
/userapps/rikuhut/.local/bin/hatch run python src/psg_gan/train.py -p $PARAM_PATH \
    --visible-device 0 --mlflow-tracking-uri ${MLFLOW_TRACKING_URI:-http://localhost:5000} \
    --mlflow-experiment ${MLFLOW_EXPERIMENT:-'psg-gan-new'}
