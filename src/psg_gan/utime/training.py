import copy
import grain
import jax
import jax.numpy as jnp
import logging
import mlflow
import numpy as np
import optax
import tempfile

from datetime import datetime as dt
from flax import nnx
from functools import partial
from orbax.checkpoint import v1 as ocp
from pathlib import Path
from psg_gan.metrics import classification
from psg_gan.utime import models
from tqdm import tqdm
from typing import Any


logger = logging.getLogger(__name__)


def get_model(model_config: dict[str, Any], rngs: nnx.Rngs):
    return models.UTime(
        input_names=model_config['input_names'],
        block_args=model_config['block_args'],
        output_args=model_config['output_args'],
        aspp_depth=model_config['aspp_depth'],
        rngs=rngs
    )


def softmax_focal_loss_with_integer_labels(
        logits: jax.Array,
        labels: jax.Array,
        gamma: float = 2.0,
        alpha: jax.Array | None = None,
        axis: int | None = -1,
        where: jax.Array | None = None) -> jax.Array:
    """Focal loss with softmax instead of sigmoid.

    Args:
        gamma : if 0.0 -> categorical crossentropy
    """
    _msg = "Alpha shape should equal number of classes."
    assert alpha is None or alpha.shape[0] == logits.shape[axis], _msg
    p = jax.nn.softmax(logits, axis=axis)
    one_hot = jax.nn.one_hot(labels, logits.shape[axis])
    ce_loss = -(one_hot * jax.nn.log_softmax(logits, axis, where))
    focal_factor = (1.0 - p) ** gamma
    if alpha is not None:
        focal_factor = alpha * focal_factor

    return (ce_loss * focal_factor).sum(axis=axis, where=where)


# UPDATE: use train_step_func = nnx.jit(partial(train_step, loss_params=...)) instead of @partial(nnx.jit, ...)
#@partial(nnx.jit, static_argnames=('focal_loss_gamma',))
def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        ema: jax.Array,
        inputs: dict[str, jax.Array],
        outputs: dict[str, jax.Array],
        loss_params: dict[str, Any]):
    def loss_fn(model):
        logits = model(inputs)
        aux_losses = {}
        for oname in logits.keys():
            #_loss = optax.softmax_cross_entropy_with_integer_labels(logits[oname], outputs[oname]).mean()
            _loss = softmax_focal_loss_with_integer_labels(
                logits[oname], outputs[oname],
                gamma=loss_params[oname]['gamma'],
                alpha=loss_params[oname]['alpha']).mean()
            aux_losses[f'{oname}_loss'] = _loss

        loss = sum(aux_losses.values())

        return loss, (logits, aux_losses)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, aux_losses)), grads = grad_fn(model)
    optimizer.update(model, grads)
    ema.update(model)

    accuracies = {}
    for oname in outputs.keys():
        _acc = jnp.mean(
            jnp.argmax(logits[oname], -1) == outputs[oname]
        )
        accuracies[f'{oname}_accuracy'] = _acc

    metrics = {'loss': loss, **aux_losses, **accuracies}

    return metrics


def eval_step(
        model: nnx.Module,
        inputs: dict[str, jax.Array],
        outputs: dict[str, jax.Array],
        loss_params: dict[str, Any]) -> dict[str, float]:
    logits = model(inputs)

    # When padding the data to avoid XLA recompilations with variable length recordings,
    # the outputs should be padded with -1.
    # TODO: figure out how to implement masking for the models and loss function also.
    losses = {}
    accuracies = {}
    for oname in outputs.keys():
        _outs = outputs[oname]
        mask = _outs != -1
        _outs = jnp.where(mask, _outs, jnp.zeros_like(_outs))
        _loss = jnp.mean(
            softmax_focal_loss_with_integer_labels(
                logits[oname], _outs,
                gamma=loss_params[oname]['gamma'],
                alpha=loss_params[oname]['alpha']),
            where=mask
        )
        losses[f'{oname}_loss'] = _loss

        _acc = jnp.mean(
            jnp.argmax(logits[oname], -1) == _outs, where=mask
        )
        accuracies[f'{oname}_accuracy'] = _acc

    loss = sum(losses.values())

    return {'loss': loss, **losses, **accuracies}


def train_epoch(train_step_func, model, optimizer, ema, train_iter, steps):
    metrics = []

    for _, (inps, outs) in tqdm(zip(range(steps), train_iter), leave=False):
        _metrics = train_step_func(model, optimizer, ema, inps, outs)
        metrics.append(_metrics)

    metrics_dict = {k: [d[k] for d in metrics] for k in metrics[0]}
    mean_metrics = {k: np.mean(metrics_dict[k]) for k in metrics_dict}

    return mean_metrics


def get_learning_rate_fn(transition_steps=100,
                         peak_value=0.1,
                         pct_start=0.3,
                         pct_final=0.8,
                         div_factor=10.0,
                         final_div_factor=1e2):
    lr_fn = optax.linear_onecycle_schedule(
        transition_steps=transition_steps,
        peak_value=peak_value,
        pct_start=pct_start,
        pct_final=pct_final,
        div_factor=div_factor,
        final_div_factor=final_div_factor)
    return lr_fn


@optax.inject_hyperparams
def get_optax_optimizer(learning_rate):
    # By using optax.inject_hyperparams, the current value of learning rate
    # will be available in opt_state.hyperparams during training.
    # This is useful for logging in the training loop.
    return optax.yogi(learning_rate=learning_rate)
    #return optax.adamw(learning_rate=learning_rate)


def get_optimizer(model, train_config, train_size):
    steps_per_epoch = train_config.get('steps_per_epoch') or (train_size // train_config['batch_size'])
    transition_steps = train_config['epochs'] * steps_per_epoch
    learning_rate_fn = get_learning_rate_fn(
        transition_steps=transition_steps,
        peak_value=train_config['peak_learning_rate'],
        div_factor=train_config['lr_div_factor'],
        final_div_factor=train_config['lr_final_div_factor'])
    
    optax_optimizer = get_optax_optimizer(learning_rate_fn)
    optimizer = nnx.Optimizer(model, optax_optimizer, wrt=nnx.Param)

    return optimizer


def restore_checkpoint(
        ckptr: ocp.training.Checkpointer,
        step: int = None,
        restore_ema: bool = False,
        cpu: bool = False) -> nnx.Module:
    # First, only load the configs
    cfgs = ckptr.load_checkpointables(step, {'model_config':None, 'train_config': None})

    # Create the abstract model for restoring
    rngs = nnx.Rngs(cfgs['train_config']['seed'])
    abstract_model = nnx.eval_shape(lambda rngs: get_model(cfgs['model_config'], rngs), rngs)
    graphdef, abstract_state = nnx.split(abstract_model)

    # Change sharding to work if restored in different topology
    if cpu:
        devices = jax.devices('cpu')
    else:
        devices = jax.devices()

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, ('model_state',)),
        jax.sharding.PartitionSpec(),
    )
    def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        return x.update(sharding=sharding)

    abstract_state = jax.tree_util.tree_map(set_sharding, abstract_state)

    # Create the actual model from the restored state
    if restore_ema:
        restored = ckptr.load_checkpointables(step, {'ema_model_state': abstract_state})
        model = nnx.merge(graphdef, restored['ema_model_state'])
    else:
        restored = ckptr.load_checkpointables(step, {'model_state': abstract_state})
        model = nnx.merge(graphdef, restored['model_state'])
    model.eval()

    return {'model': model, **cfgs}


def dict_to_io(d, input_names=[], output_names=[]):
    inputs = {k: d[k] for k in input_names}
    outputs = {k: d[k] for k in output_names}

    return inputs, outputs


def training_loop(
        train_config: dict[str, Any],
        model_config: dict[str, Any],
        train_datasets: dict[str, grain.MapDataset],
        val_datasets: dict[str, grain.MapDataset]) -> nnx.Module:
    logger.info('Start training loop...')
    train_start = dt.now().strftime('%Y%m%d_%H%M%S')

    ckpt_root_dir = train_config.get('ckpt_root_dir') or '/tmp'
    run_name = mlflow.active_run().info.run_name
    ckpt_dir = Path(ckpt_root_dir) / run_name
    mlflow.log_param('ckpt_dir', str(ckpt_dir))

    io_map_func = partial(dict_to_io,
        input_names=model_config['input_names'],
        output_names=model_config['output_names'])

    # Randomly sample from each training set
    train_size = sum([len(ds) for ds in train_datasets.values()])
    train_ds = grain.MapDataset.mix(
        train_datasets.values(),
        weights=[len(ds) / train_size for ds in train_datasets.values()],
    )
    train_ds = (train_ds
        .seed(train_config['seed'])
        .shuffle()
        .repeat()
        .map(io_map_func)
        .batch(train_config['batch_size'], drop_remainder=True)
        .to_iter_dataset())

    train_ds = (train_ds
        .mp_prefetch(grain.MultiprocessingOptions(
            num_workers=train_config['grain_num_workers'],
            per_worker_buffer_size=train_config['grain_per_worker_buffer_size']))
        .map(jax.device_put)
    )
    train_iter = iter(train_ds)

    val_datasets = {
        k: val_datasets[k].map(io_map_func)
        for k in val_datasets.keys()
    }

    rngs = nnx.Rngs(train_config['seed'])
    model = get_model(model_config, rngs)
    optimizer = get_optimizer(model, train_config, train_size)

    ema = nnx.EMA(model, decay=train_config['ema_decay'])
    ema_model = ema.apply_to(model)

    best_epoch = 0
    best_val_loss = np.inf

    steps_per_epoch = train_config.get('steps_per_epoch') or (train_size // train_config['batch_size'])

    # Cast the focal loss alpha to jax.Array for the loss.
    # Deep copy the params to avoid checkpointing issues with train_config
    _loss_params = copy.deepcopy(train_config['loss_params'])
    for oname in _loss_params:
        _loss_params[oname]['alpha'] = jnp.array(_loss_params[oname]['alpha'])

    # JIT the train and eval steps here so that the compilation cache works correctly
    # with partial functions
    # https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching
    _jitted_train_step = nnx.jit(partial(train_step, loss_params=_loss_params))
    _jitted_eval_step = nnx.jit(partial(eval_step, loss_params=_loss_params))

    with ocp.training.Checkpointer(
            ckpt_dir,
            preservation_policy=ocp.training.preservation_policies.LatestN(3),
            custom_metadata={'train_start': train_start}) as ckptr:
        for epoch in tqdm(range(1, train_config['epochs'] + 1)):
            model.train()
            train_metrics = train_epoch(_jitted_train_step, model, optimizer, ema, train_iter, steps_per_epoch)

            logger.info(f'epoch: {epoch}, train_metrics: {str(train_metrics)}')
            mlflow.log_metric('epoch', epoch)

            for k, v in train_metrics.items():
                mlflow.log_metric(f'train_{k}', v, step=epoch)

            all_val_losses = {}
            for val_key, val_ds in val_datasets.items():
                val_metrics = []
                ema_model.eval()
                _val_iter = iter(val_ds.batch(1))
                for inps, outs in tqdm(_val_iter):
                    val_metrics.append(_jitted_eval_step(ema_model, inps, outs))

                val_metrics_dict = {k: [d[k] for d in val_metrics] for k in val_metrics[0]}
                mean_val_metrics = {k: np.mean(val_metrics_dict[k]) for k in val_metrics_dict}
                all_val_losses[val_key] = mean_val_metrics['loss']
                
                logger.info(f'epoch: {epoch}, {val_key}_metrics: {str(mean_val_metrics)}')
                for k, v in mean_val_metrics.items():
                    mlflow.log_metric(f'{val_key}_{k}', v, step=epoch)
                    
            mean_val_loss = np.mean(list(all_val_losses.values()))
            mlflow.log_metric('mean_val_loss', mean_val_loss, step=epoch)

            if mean_val_loss < best_val_loss:
                logger.info('Saving new best model...')
                best_val_loss = mean_val_loss
                best_epoch = epoch
                ckptr.save_checkpointables_async(step=epoch, checkpointables={
                    'model_state':  nnx.state(model),
                    'ema_model_state':  nnx.state(ema_model),
                    'model_config': model_config,
                    'train_config': train_config
                })
 
        # Log the best epoch after training
        mlflow.log_metric('best_val_loss', best_val_loss)
        mlflow.log_metric('best_epoch', best_epoch)

        # Restore the model parameters on CPU device, since recompilation due to different length recordings
        # during testing with full-night PSGS seems to be three times slower on GPU.
        # TODO: investigate why this is the case
        best_model = restore_checkpoint(ckptr, restore_ema=True, cpu=True)['model']

    return best_model


def evaluate(
        model: nnx.Module,
        ds: grain.MapDataset,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        prefix: str = 'test',
        cpu: bool = True) -> None:
    logger.info('Evaluating test set...')

    io_map_func = partial(dict_to_io,
        input_names=model_config['input_names'],
        output_names=model_config['output_names'])

    # Use padded_batch to avoid XLA recompilation on every sample...
    # TODO: calling models on padded inputs drastically change the outputs,
    # leading to worse classification performance. Possible reasons:
    # - Operations that use global information, e.g. SE, batchnorm
    ds_iter = (ds
        .map(io_map_func)
        # .padded_batch(16, padding_values=(
        #     {k: 0.0 for k in model_config['input_names']},
        #     {k: -1 for k in model_config['output_names']}))
        # .unbatch()
        .batch(1)
        .to_iter_dataset()
    )
    # Load the data to CPU instead of GPU if cpu=True
    if cpu:
        device = jax.devices('cpu')[0]
    else:
        device = jax.devices()[0]
    ds_iter = (ds_iter
        .mp_prefetch(grain.MultiprocessingOptions(
            num_workers=2,
            per_worker_buffer_size=5))
        .map(partial(jax.device_put, device=device))
    )

    model.eval()
    logger.info(f'Computing outputs for {prefix}...')
    all_outputs = {k: [] for k in model_config['output_names']}
    all_logits = {k: [] for k in model_config['output_names']}
    for inps, outs in tqdm(ds_iter):
        logits = model(inps)
        for k in outs.keys():
            all_outputs[k].append(outs[k][0])
            all_logits[k].append(logits[k][0])

    labels_flat = {k: np.concatenate(v) for k, v in all_outputs.items()}
    logits_flat = {k: np.concatenate(v) for k, v in all_logits.items()}

    logger.info(f'Computing metrics for {prefix}...')
    mean_losses = {
        k: np.mean(softmax_focal_loss_with_integer_labels(
            logits_flat[k], labels_flat[k],
            gamma=train_config['loss_params'][k]['gamma'],
            alpha=jnp.array(train_config['loss_params'][k]['alpha'])))
        for k in outs.keys()
    }
    mean_accuracies = {
        k: np.mean(np.argmax(logits_flat[k], -1) == labels_flat[k])
        for k in outs.keys()
    }

    for k, v in mean_losses.items():
        mlflow.log_metric(f'{prefix}_{k}_loss', v)
    
    for k, v in mean_accuracies.items():
        mlflow.log_metric(f'{prefix}_{k}_accuracy', v)

    # Calculate confusion matrices and classification metrics for each output and log to mlflow
    plot_kwargs = {
        'hypnogram': {
            'class_names': ['W', 'N1', 'N2', 'N3', 'R'],
            'cm_figsize': (7, 6)
        },
        'events': {
            'class_names': ['No event', 'Hypopnea', 'Apnea'],
            'cm_figsize': (5, 4.5)
        },
    }
    logger.info(f'Running classification.plot_metrics() for {prefix}...')
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for k in outs.keys():
            pred_labels = np.argmax(logits_flat[k], -1)
            save_prefix = f'{prefix}_{k}'
            classification.plot_metrics(
                true=labels_flat[k],
                pred=pred_labels,
                class_names=plot_kwargs[k]['class_names'],
                save_dir=temp_dir,
                save_prefix=save_prefix,
                cm_xlabel='Automatic',
                cm_ylabel='Manual',
                cm_figsize=plot_kwargs[k]['cm_figsize'])
            mlflow.log_artifact(temp_dir / f'{save_prefix}_cm.pdf')
            mlflow.log_artifact(temp_dir / f'{save_prefix}_clf_report.json')
