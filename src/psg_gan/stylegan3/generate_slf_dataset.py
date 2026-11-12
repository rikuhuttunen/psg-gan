import argparse
import jax
import jax.numpy as jnp
import logging
import numpy as np
import psg_gan.stylegan3.training as stylegan3_training
import psg_gan.utime.training as utime_training
import sleeplab_format as slf

from datetime import datetime as dt
from datetime import timedelta
from orbax.checkpoint import v1 as ocp
from pathlib import Path
from psg_gan.utime import scorer
from tqdm import tqdm


logger = logging.getLogger(__name__)


def arr_to_dict(arr: jax.Array, keys: list[str]) -> dict[str, jax.Array]:
    return {k: arr[...,i][..., jnp.newaxis] for i, k in enumerate(keys)}


def parse_sample_arrays(x_gs, arr_names, fs, start_ts) -> dict[str, slf.models.SampleArray]:
    sample_arrays = {}
    for i, name in enumerate(arr_names):
        attributes = slf.models.ArrayAttributes(
            name=name,
            start_ts=start_ts,
            sampling_rate=float(fs)
        )
        values_func = lambda bound_s=x_gs[..., i]: bound_s
        sample_arrays[name] = slf.models.SampleArray(attributes=attributes, values_func=values_func)

    return sample_arrays


def to_slf_subject(subject_id, x_gs, scorer_logits, gan_config,
                   start_ts=dt.utcfromtimestamp(0),
                   scorer_name='utime') -> slf.models.Subject:
    sample_arrays = parse_sample_arrays(x_gs, gan_config['input_names'], gan_config['fs'], start_ts)
    
    if scorer_logits is not None:
        scorer_outputs = {k: jax.nn.softmax(v) for k, v in scorer_logits.items()}
        annotations = scorer.parse_annotations(scorer_outputs, start_ts, scorer=scorer_name)
    else:
        annotations = None

    metadata = slf.models.SubjectMetadata(subject_id=subject_id, recording_start_ts=start_ts)

    return slf.models.Subject(
        metadata=metadata,
        sample_arrays=sample_arrays,
        annotations=annotations
    )


def generate_slf_ds(
        nsamples: int,
        batch_size: int,
        gan_ckpt_dir: Path,
        scorer_ckpt_dir: Path,
        ds_name: str,
        series_name: str,
        scorer_name: str,
        rng_seed: int = 42) -> slf.models.Dataset:
    """Synthesize data with GAN and score with automatic model."""
    assert nsamples % batch_size == 0
    nbatches = nsamples // batch_size

    rng = jax.random.PRNGKey(rng_seed)
    with ocp.training.Checkpointer(gan_ckpt_dir) as ckptr:
        gan_ckpt = stylegan3_training.restore_checkpoint(ckptr, restore_ema=True)

    if scorer_ckpt_dir == '':
        logger.info('Automatic scoring is turned off, since scorer_ckpt_dir was not given...')
        scorer_ckpt = None
    else:
        with ocp.training.Checkpointer(scorer_ckpt_dir) as ckptr:
            scorer_ckpt = utime_training.restore_checkpoint(ckptr, restore_ema=True)

    subjects = {}

    for i in tqdm(range(nbatches)):
        rng, z_key = jax.random.split(rng, 2)
        zs = jax.random.normal(z_key, [batch_size, gan_ckpt['model_config']['z_dim']])
        x_gs = gan_ckpt['model'](zs)
        
        if scorer_ckpt is not None:
            input_dict = arr_to_dict(x_gs, scorer_ckpt['model_config']['input_names'])
            scorer_logits = scorer_ckpt['model'](input_dict)
        
        for j in tqdm(range(batch_size), leave=False):
            subject_id = str(i*batch_size + j)
            
            if scorer_ckpt is not None:
                _scorer_logits = {k: np.array(v[j])[np.newaxis, ...] for k, v in scorer_logits.items()}
            else:
                _scorer_logits = None

            subjects[subject_id] = to_slf_subject(
                subject_id,
                np.array(x_gs[j, ...]),
                _scorer_logits,
                gan_config=gan_ckpt['model_config'],
                scorer_name=scorer_name)

    series = slf.models.Series(name=series_name, subjects=subjects)
    ds = slf.models.Dataset(name=ds_name, series={series_name: series})    
    return ds


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nsamples', type=int, required=True,
        help='the number of samples to be generated')
    parser.add_argument('--ds-base-dir', type=Path, required=True,
        help='Folder where the SLF dataset is saved')
    parser.add_argument('--ds-name', type=str, required=True,
        help='The name of the created SLF dataset')
    parser.add_argument('--series-name', type=str, required=True,
        help='The name of the created SLF series') 
    parser.add_argument('--gan-ckpt-dir', type=Path, required=True,
        help='Path to the GAN checkpoint')
    parser.add_argument('--scorer-ckpt-dir', type=Path, default='',
        help='Path to the automatic scoring model checkpoint. Empty string == no scoring.')
    parser.add_argument('--scorer-name', type=str, default='utime')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
        help='batch size for generating and scoring samples')

    return parser


def run_cli() -> None:
    parser = get_parser()
    args = parser.parse_args()

    slf_ds = generate_slf_ds(
        args.nsamples,
        args.batch_size,
        args.gan_ckpt_dir,
        args.scorer_ckpt_dir,
        args.ds_name,
        args.series_name,
        args.scorer_name)
    
    slf.writer.write_dataset(slf_ds, basedir=args.ds_base_dir)
    logger.info('Dataset generation finished.')


if __name__ == '__main__':
    run_cli()
