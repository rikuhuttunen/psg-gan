import grain
import jax
import jax.numpy as jnp
import logging
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax

from datetime import datetime as dt
from flax import nnx
from functools import partial
from numpy.typing import ArrayLike
from orbax.checkpoint import v1 as ocp
from pathlib import Path
from typing import Any, Callable
from psg_gan.metrics.fud import calculate_fud
from stylegan3_1d_flax.stylegan2.models import Discriminator
from stylegan3_1d_flax.stylegan3.models import Generator


logger = logging.getLogger(__name__)


def get_G(model_config: dict[str, Any]) -> nnx.Module:
    return Generator(
        width_sec=model_config['width_sec'],
        output_fs=model_config['fs'],
        output_channels=len(model_config['input_names']),
        z_dim=model_config['z_dim'],
        w_dim=model_config['w_dim'],
        num_mapping_layers=model_config['num_mapping_layers'],
        mapping_lr_multiplier=model_config['mapping_lr_multiplier'],
        channel_base=model_config['channel_base'],
        channel_max=model_config['channel_max'],
        num_synthesis_layers=model_config['num_synthesis_layers'],
        num_critical=model_config['num_critical'],
        first_cutoff=model_config['first_cutoff'],
        first_stopband=model_config['first_stopband'],
        last_stopband_rel=model_config['last_stopband_rel'],
        margin_size=model_config['margin_size'],
        output_scale=model_config['output_scale'],
        conv_ksize=model_config['conv_ksize'],
        last_conv_ksize=model_config['last_conv_ksize'],
        filter_size=model_config['filter_size'],
        lrelu_upsampling=model_config['lrelu_upsampling'],
        conv_clamp=model_config['conv_clamp'],
        magnitude_ema_beta=model_config['magnitude_ema_beta'],
        rngs=nnx.Rngs(model_config['rng_seed'])
    )


def get_D(model_config: dict[str, Any]) -> nnx.Module:
    return Discriminator(
        input_channels=len(model_config['input_names']),
        input_width=model_config['width_sec'] * model_config['fs'],
        block_args=model_config['d_block_args'],
        mbstd_group_size=model_config['mbstd_group_size'],
        mbstd_num_new_features=model_config['mbstd_num_new_features'],
        rngs=nnx.Rngs(model_config['rng_seed'])
    )


@nnx.jit
def g_train_step(g_model, d_model, g_optimizer, z):
    def loss_fn(g_model):
        x_fake = g_model(z, update_emas=True)
        d_logits = d_model(x_fake)
        loss = jnp.mean(jax.nn.softplus(-d_logits))  # -log(sigmoid(d_logits))

        return loss

    logger.info('g_train_step tracing...')

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(g_model)
    g_optimizer.update(g_model, grads)
    metrics = {'g_loss': loss}

    return metrics


@partial(nnx.jit, static_argnames=('r1_gamma', 'regularize'))
def d_train_step(g_model, d_model, d_optimizer, z, x_real, r1_gamma=10.0, regularize=False):
    def loss_fn(d_model):
        x_fake = g_model(z, update_emas=False)
        d_real = d_model(x_real)
        d_fake = d_model(x_fake)

        log_loss = jax.nn.softplus(d_fake)  # -log(1-sigmoid(d_fake))
        log_loss += jax.nn.softplus(-d_real)  # -log(sigmoid(d_real))

        return jnp.mean(log_loss)

    def r1_loss_fn(d_model):
        real_grad_fn = nnx.grad(
            lambda x: jnp.sum(d_model(x))
        )
        real_grads = real_grad_fn(x_real)
        r1_penalty = jnp.sum(jnp.square(real_grads), axis=[1, 2])
        r1_loss = r1_penalty * (r1_gamma / 2)

        return jnp.mean(r1_loss)

    logger.info(f'd_train_step tracing with regularize={regularize}')

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(d_model)
    d_optimizer.update(d_model, grads)
    metrics = {'d_loss': loss}

    if regularize:
        r1_grad_fn = nnx.value_and_grad(r1_loss_fn)
        r1_loss, r1_grads = r1_grad_fn(d_model)
        d_optimizer.update(d_model, r1_grads)
        metrics['r1_loss'] = r1_loss

    return metrics


def get_lr_fn(
        init_value: float,
        peak_value: float,
        warmup_steps: int,
        decay_steps: int,
        end_value: float = 0.0,
        exponent: float = 1.0) -> Callable:
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
        exponent=exponent
    )


def get_optimizer(model, opt_cfg, total_steps):
    reg_interval = opt_cfg.get('reg_interval') or None
    if reg_interval is not None:
        reg_coeff = reg_interval / (reg_interval + 1)
    else:
        reg_coeff = 1.0

    lr_fn = get_lr_fn(
        init_value=opt_cfg['init_lr'] * reg_coeff,
        peak_value=opt_cfg['peak_lr'] * reg_coeff,
        warmup_steps=opt_cfg['warmup_steps'],
        decay_steps=total_steps,
        end_value=opt_cfg['end_lr'],
        exponent=opt_cfg.get('exponent') or 1.0
    )
    optax_optimizer = optax.adam(
        learning_rate=lr_fn,
        b1=opt_cfg['beta1'] ** reg_coeff,
        b2=opt_cfg['beta2'] ** reg_coeff
    )

    return nnx.Optimizer(model, optax_optimizer, wrt=nnx.Param)


def restore_checkpoint(
        ckptr: ocp.training.Checkpointer,
        step: int | None = None,
        restore_ema: bool = False) -> dict[str, Any]:
    cfgs = ckptr.load_checkpointables(step, {'model_config': None, 'train_config': None})

    # Create the abstract model for restore args
    abstract_model = nnx.eval_shape(lambda: get_G(cfgs['model_config']))
    graphdef, abstract_state = nnx.split(abstract_model)

    # Change sharding to work if restored in different topology
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ('model_states')),
        jax.sharding.PartitionSpec(),
    )
    def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        return x.update(sharding=sharding)

    abstract_state = jax.tree_util.tree_map(set_sharding, abstract_state)

    # Restore the model state
    restored = ckptr.load_checkpointables(
        step,
        {'model_state': abstract_state, 'model_state_ema': abstract_state})

    # Create the actual model from the restored state
    if restore_ema:
        model = nnx.merge(graphdef, restored['model_state_ema'])
    else:
        model = nnx.merge(graphdef, restored['model_state'])
    model.eval()

    return {'model': model, **cfgs}


def configure_ds(
        ds: grain.MapDataset,
        train_config: dict[str, Any],
        model_config: dict[str, Any],
        crop_size: int | None = None) -> grain.IterDataset[Any]:
    def _crop(s: Any, rng: np.random.Generator) -> ArrayLike:
        _shape = s.shape
        _start_idx = rng.integers(0, _shape[0] - crop_size + 1)
        return s[_start_idx:_start_idx+crop_size, ...]

    _ds = (ds
        .seed(train_config['seed'])
        .map(lambda d: np.concat([d[k] for k in model_config['input_names']], -1))
    )
    if crop_size is not None:
        _ds = _ds.random_map(_crop)

    _ds = (_ds
        .shuffle()
        .repeat()
        .batch(train_config['batch_size'], drop_remainder=True)
        .to_iter_dataset()
    )
    _ds = (_ds
        .mp_prefetch(grain.MultiprocessingOptions(
            num_workers=train_config['grain_num_workers'],
            per_worker_buffer_size=train_config['grain_per_worker_buffer_size']
        ))
        .map(jax.device_put)
    )

    return _ds


def training_loop(
        train_config: dict[str, Any],
        model_config: dict[str, Any],
        train_datasets: dict[str, grain.MapDataset]) -> None:
    train_ds = train_datasets['train']  # Pick only the 'train' split for now...
    total_samples = train_config['total_samples']
    logger.info(f'Start training StyleGAN3 for {total_samples} samples...')
    train_start = dt.now().strftime('%Y%m%d_%H%M%S')

    real_iter = iter(configure_ds(train_ds, train_config, model_config))

    g_model = get_G(model_config)
    g_optimizer = get_optimizer(
        g_model, train_config['g_opt'], total_steps=total_samples // train_config['batch_size'])

    # TODO: Consider if using nnx.EMA would be better
    g_ema_def, g_state = nnx.split(g_model)
    g_ema = optax.ema(decay=train_config['g_ema_decay'])
    g_ema_state = g_ema.init(g_state)

    d_model = get_D(model_config)
    d_optimizer = get_optimizer(
        d_model, train_config['d_opt'], total_steps=total_samples // train_config['batch_size'])

    # Log real examples
    x_real = next(real_iter)
    log_examples(x_real[:2, ...], 0, total_samples, model_config['input_names'], 'real')

    best_fud = np.inf
    best_step = 0
    ckpt_root_dir = train_config['ckpt_root_dir']
    if ckpt_root_dir is None:
        ckpt_root_dir = '/tmp'
    
    run_name = mlflow.active_run().info.run_name
    ckpt_dir = Path(ckpt_root_dir) / run_name
    mlflow.log_param('ckpt_dir', ckpt_dir)

    fud_stats_cache_path = Path(train_config['fud_stats_cache_path'])
    # Remove real FUD cache path so that previous cache file does not mess up things
    fud_stats_cache_path.unlink(missing_ok=True)

    cur_samples = 0
    metric_tick_samples = 0
    snapshot_tick_samples = 0
    fud_tick_samples = 0

    key = jax.random.key(train_config['seed'])
    z_key, z_example_key, log_fud_key = jax.random.split(key, 3)

    batch_size = train_config['batch_size']
    reg_interval = train_config['d_opt']['reg_interval']

    with ocp.training.Checkpointer(
            ckpt_dir,
            preservation_policy=ocp.training.preservation_policies.LatestN(3),
            custom_metadata={'train_start': train_start}) as ckptr:
        while cur_samples < total_samples:
            cur_batch = cur_samples // batch_size
            cur_samples += batch_size

            x_real = next(real_iter)
            z_key, z_d_key, z_g_key = jax.random.split(z_key, 3)
            z_d = jax.random.normal(z_d_key, (x_real.shape[0], model_config['z_dim']))
            z_g = jax.random.normal(z_g_key, (x_real.shape[0], model_config['z_dim']))

            regularize = cur_batch % reg_interval == 0
            d_metrics = d_train_step(g_model, d_model, d_optimizer, z_d, x_real,
                                    r1_gamma=train_config['r1_gamma'] * reg_interval,
                                    regularize=regularize)
            
            g_metrics = g_train_step(g_model, d_model, g_optimizer, z_g)

            _, g_state = nnx.split(g_model)
            _, g_ema_state = g_ema.update(g_state, g_ema_state)

            if cur_samples > metric_tick_samples + train_config['log_metrics_interval_samples']:
                logger.info(f'Current samples {cur_samples}/{total_samples}: logging metrics...')
                metric_tick_samples = cur_samples
                mlflow.log_metric('cur_samples', cur_samples)
                log_metrics(d_metrics, cur_samples)
                log_metrics(g_metrics, cur_samples)

            if cur_samples > snapshot_tick_samples + train_config['snapshot_interval_samples']:
                logger.info(f'Current samples {cur_samples}/{total_samples}: logging examples...')
                snapshot_tick_samples = cur_samples
                z_example_key, _key = jax.random.split(z_example_key)
                z_examples = jax.random.normal(_key, (2, model_config['z_dim']))

                x_g = g_model(z_examples)
                log_examples(x_g, cur_samples, total_samples, model_config['input_names'], 'generated')
                g_ema_model = nnx.merge(g_ema_def, g_ema_state.ema)

                x_g_ema = g_ema_model(z_examples)
                log_examples(x_g_ema, cur_samples, total_samples, model_config['input_names'], 'generated_ema')

            if cur_samples > fud_tick_samples + train_config['log_fud_interval_samples']:
                fud_tick_samples = cur_samples

                # Compute FUD with 5min random segments
                fud_crop_size = 5 * 60 * model_config['fs']
                fud = log_fud(cur_samples, g_model, log_fud_key, train_ds, train_config, model_config,
                            train_config['n_fud_real'], train_config['n_fud_fake'], train_config['utime_ckpt_dir'],
                            fud_stats_cache_path, crop_size=fud_crop_size)
                logger.info(f'FUD = {fud}')

                if fud < best_fud:
                    logger.info(f'Saving generator with FUD = {fud}...')
                    best_fud = fud
                    best_step = cur_samples
                    g_ema_model = nnx.merge(g_ema_def, g_ema_state.ema)
                    ckptr.save_checkpointables_async(step=cur_samples, checkpointables={
                        'model_state': nnx.state(g_model),
                        'model_state_ema': nnx.state(g_ema_model),
                        'model_config': model_config,
                        'train_config': train_config
                    })

    mlflow.log_metric('best_fud', best_fud, step=best_step)

    return None


def log_fud(cur_samples, g_model, g_iter_key, train_ds, train_config, model_config, n_real, n_fake,
            utime_ckpt_dir, fud_stats_cache_path, crop_size=None, batch_size=8) -> float:
    def iterate_g(batch_size, key):
        def _crop(key, s):
            _shape = jnp.shape(s)
            start_idx = jax.random.randint(key, shape=(), minval=0, maxval=_shape[0] - crop_size + 1)
            return jax.lax.dynamic_slice(s, (start_idx, 0), (crop_size, _shape[-1]))

        @nnx.jit
        def generate(g, z):
            return g(z)

        while True:
            key, z_key, crop_key = jax.random.split(key, 3)
            z = jax.random.normal(z_key, (batch_size, model_config['z_dim']))
            x_g = generate(g_model, z)
            if crop_size is not None:
                _keys = jax.random.split(crop_key, batch_size)
                x_g = jax.vmap(_crop)(_keys, x_g)

            yield x_g

    _train_config = train_config.copy()
    _train_config.update({'batch_size': batch_size})
    real_iter = iter(configure_ds(train_ds, _train_config, model_config, crop_size=crop_size))
    fake_iter = iterate_g(batch_size, g_iter_key)
    fud = calculate_fud(real_iter, fake_iter, n_real, n_fake,
                        batch_size, Path(utime_ckpt_dir), fud_stats_cache_path)
    mlflow.log_metric('fud', fud, step=cur_samples)

    return fud


def log_examples(x, cur_samples, total_samples, input_names, fname_prefix,
                 zoom_dur_sec=60, zoom_start_sec=5*60, fs=64):
    samples_digits = len(str(total_samples))
    cur_samples_str = str(cur_samples).zfill(samples_digits)

    zoom_start_idx = int(zoom_start_sec * fs)
    zoom_end_idx = int(zoom_start_idx + fs * zoom_dur_sec)

    nsamples = x.shape[0]
    nchannels = x.shape[-1]

    fig, axs = plt.subplots(nsamples*nchannels, 2, figsize=(12, 2*nsamples))

    for i in range(nsamples):
        for ch in range(nchannels):
            rowidx = i * nchannels + ch
            axs[rowidx, 0].plot(x[i, :, ch])
            axs[rowidx, 1].plot(x[i, zoom_start_idx:zoom_end_idx, ch])
            axs[rowidx, 0].set_title(input_names[ch])

    mlflow.log_figure(fig, f'{fname_prefix}_{cur_samples_str}.png')
    plt.close(fig)


def log_metrics(mdict: dict[str, int|float], cur_samples: int) -> None:
    for k, v in mdict.items():
        mlflow.log_metric(k, v, step=cur_samples)
