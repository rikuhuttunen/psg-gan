import jax.numpy as jnp
import logging
import numpy as np
import pickle
import psg_gan.utime.training as utime_training
import scipy

from flax import nnx
from orbax.checkpoint import v1 as ocp
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


def compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake):
    m = np.square(mu_fake - mu_real).sum()
    s = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real))
    dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
    return np.real(dist)


@nnx.jit
def get_aspp_features(inputs, model):
    feats = model.aspp(model.encoder(inputs)[0])
    
    return feats


def calculate_fud(
        real_iter: Iterable,
        fake_iter: Iterable,
        real_ns: int,
        fake_ns: int,
        batch_size: int,
        utime_ckpt_dir: Path,
        real_statistics_cache_path: Path | None = None) -> list[float]:
    with ocp.training.Checkpointer(utime_ckpt_dir) as ckptr:
        model = utime_training.restore_checkpoint(ckptr, restore_ema=True)['model']
    model.eval()

    real_feats = []
    fake_feats = []

    if (real_statistics_cache_path is not None) and (real_statistics_cache_path.exists()):
        logger.info('Loading real statistics from cache...')
        with open(real_statistics_cache_path, 'rb') as f:
            mu_real, sigma_real = pickle.load(f, encoding='latin1')
    else:
        logger.info('Computing mu and sigma for real samples...')
        nbatches = real_ns // batch_size
        for _ in range(nbatches):
            real = next(real_iter)
            batch_feats = get_aspp_features(real, model)

            # Flatten the features, keep batch dimension
            real_feats.append(jnp.reshape(batch_feats, [jnp.shape(batch_feats)[0], -1]))

        real_feats = np.concatenate(real_feats, axis=0)
        mu_real = np.mean(real_feats, axis=0)
        sigma_real = np.cov(real_feats, rowvar=False)

        if real_statistics_cache_path is not None:
            with open(real_statistics_cache_path, 'wb') as f:
                pickle.dump((mu_real, sigma_real), f, protocol=pickle.HIGHEST_PROTOCOL) 

    logger.info('Computing mu and sigma for fake samples...')
    nbatches = fake_ns // batch_size
    for _ in range(nbatches):
        fake = next(fake_iter)
        batch_feats = get_aspp_features(fake, model)
        fake_feats.append(jnp.reshape(batch_feats, [jnp.shape(batch_feats)[0], -1]))

    fake_feats = np.concatenate(fake_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_fake = np.cov(fake_feats, rowvar=False)

    logger.info('Computing Frechet distance...')
    dist = compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return dist
