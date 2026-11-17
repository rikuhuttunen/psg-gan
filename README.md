# PSG-GAN

Code related to the study described in the manuscript by Huttunen et. al. "From real to synthetic: PSG-GAN learns sleep apnea signal dynamics for AI training".

## System requirements

The model training and analyses were performed on a [SLURM](https://slurm.schedmd.com/overview.html) cluster running Rocky Linux 9 provided by the UEF Bioinformatics Center. The training was performed on a single node with an NVIDIA A100 40 GB GPU. Python 3.12 was used.

## Installation

Go to the directory where you want to install and create/activate the Python 3.12 environment e.g. with venv or conda. Then, clone the repository and install with pip:
```console
git clone https://github.com/rikuhuttunen/psg-gan.git
cd psg-gan
pip install -e .
```

## Preparing the datasets

The [STAGES](https://sleepdata.org/datasets/stages) and [MESA](https://sleepdata.org/datasets/mesa) datasets can be obtained through the National Sleep Research Resource. There are scripts to convert the MESA and STAGES datasets to sleeplab-format in https://github.com/UEF-SmartSleepLab/sleeplab-converters. 

The STAR-SA and STAR-HV datasets are currently under the process of making them publicly available.

The codes expect the data to be in [sleeplab-format](https://github.com/UEF-SmartSleepLab/sleeplab-format). The data can be then preprocessed using the config files for the sleeplab-format extractor under `configs/extractor`. E.g., the signals of interest in the STAGES dataset can be extracted and preprocessed with:
```console
slf-extract -s <STAGES SLF dir> -d <extracted data dir> -c config/extactor/STAGES_extractor_config.yml
```

## Training the automatic scoring model

Training the model used in the paper took 5 hours using a single NVIDIA A100 GPU with 40 GB memory. The models were implemented in Jax/Flax. [MLFlow](https://mlflow.org/docs/latest/ml/) was used to track model training. After the extraction of the datasets, a simultaneous scoring model for sleep stages and respiratory events can be trained with:
```console
python src/psg_gan/train.py \
    --param-path configs/utime_real.yml \
    --visible-device 0 \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment psg-gan
```

where `--visible-device` is the CUDA device numer, `mlflow-tracking-uri` is the URI for an MLFlow tracking server, and `--mlflow-experiment` is some name for the MLFlow experiment. More information on how to setup the MLFlow server can be found [in their documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/). The resulting model wil be saved in `ckpt_root_dir` defined in the config file. The `ds_dir` entries in the config file need to be changed to point to the extracted SLF datasets.

## Training the GAN

Jax/Flax implementations of StyleGAN3-based models for 1D signals are [here](https://github.com/rikuhuttunen/stylegan3-1d-flax)

Training the GAN used in the paper took 30 hours using a single NVIDIA A100 GPU with 40 GB memory. The GAN can be trained with:
```console
python src/psg_gan/train.py \
    --param-path configs/train_stylegan_c4_e1_naspres_ppg.yml \
    --visible-device 0 \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment psg-gan
```

The automatic scoring model defined in the config under `utime_ckpt_dir` is used to calculate the Frechet U-time distance (FUD) that is tracked to choose the best generator weights from GAN training. The generator weights will be saved in the `ckpt_root_dir` defined in the config file.

## Generating the synthetic dataset

The trained GAN can be used to generate a synthetic dataset in sleeplab-format, same size as in the paper, scored with the automatic model:
```console
python src/psg_gan/stylegan3/generate_slf_dataset.py \
    --nsamples 19760 \
    --batch-size 8 \
    --ds-base-dir <dir where the generated dataset is saved> \
    --ds-name <name of the created SLF dataset> \
    --series-name <name of the series created> \
    --gan-ckpt-dir <the GAN checkpoint location> \
    --scorer-ckpt-dir <the automatic scoring model checkpoint location> \
    --scorer-name <the identifier used when saving the annotations>
```

## Automatic scoring of the STAGES dataset

The STAGES dataset was automatically scored for the paper analyses. Automatic scoring can be run with:
```console
python src/psg_gan/utime/scorer.py \
    --ds-dir <the SLF dataset directory> \
    --ckpt-dir <the automatic scoring model checkpoint> \
    --scorer-name <the identifier used when saving the annotations> \
    --fs <the sampling frequency of signals used by the scoring model>
```

## Running the analyses

Python scripts for reproducing the analyses from the manuscript are provided under `src/psg_gan`.

- demographic_info.py outputs the demographic information in latex tables
- paper_plot_confmats.py was used to plot all confusion matrices used in the manuscript
- plot_training_progression.py was used to plot the FUD, generator loss and discriminator loss using a CSV exported from the MLFlow UI
- plot_subject_with_ihr.py was used to plot the example signals in the manuscript
- spectral_analysis.py was used to plot the EEG and EOG bandpowers, and sleep stage counts
- prv_frequency_analysis.py was used to the PRV bandpowers
- prv_analysis.py was used to plot the median inter-beat intervals during and after breathing obstructions, as well as statistics on counts and durations of apneas and hypopneas

## Using SLURM

There are scripts for training and using the models in a SLURM environment under `slurm/`.

## License

`psg-gan` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
