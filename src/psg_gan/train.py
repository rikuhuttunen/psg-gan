import argparse
import logging
import mlflow
import os

# Need to do this here before importing slf-grain, or logs will not be shown for some reason...
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

import slf_grain
import tempfile
import yaml

from importlib import import_module
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def set_gpu_visibility(gpu_device: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)


def log_config_to_mlflow(cfg: dict[str, Any]):
    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / 'config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(cfg, f)
            mlflow.log_artifact(f.name)


def train(config: dict[str, Any], mlflow_config: dict[str, Any]) -> None:
    """Train a model according to the config."""
    # Import the model submodule
    module = import_module(config['module_str'])

    # Instantiate the datasets
    logger.info('instantiating the datasets...')
    datasets = slf_grain.dataset.load_split_concat(config['datasets'], seed=config['training']['seed'])

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment'])

    # Run the training loop
    logger.info('running the training loop...')
    with mlflow.start_run(mlflow_config['run_id']):
        log_config_to_mlflow(config)

        # Support multiple training sets
        train_keys = [k for k in datasets.keys() if k.startswith('train')]
        train_datasets = {k: datasets[k] for k in train_keys}

        # Support multiple validation sets
        val_keys = [k for k in datasets.keys() if k.startswith('val')]
        val_datasets = {k: datasets[k] for k in val_keys}

        if len(val_datasets) > 0:
            state = module.training.training_loop(
                config['training'], config['model'], train_datasets, val_datasets)
        else:
             state = module.training.training_loop(
                 config['training'], config['model'], train_datasets
             )

        # Support multiple test sets
        test_keys = [k for k in datasets.keys() if k.startswith('test')]
        if len(test_keys) > 0:
            logger.info('Evaluating the model with test sets...')
            for test_key in test_keys:
                logger.info(f'Evaluating model on {test_key}...')
                module.training.evaluate(state, datasets[test_key], config['model'], config['training'], prefix=test_key)

    logger.info('training done.')


def get_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser to parse the cli args.
    
    A --module, or -m argument is mandatory
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--param-path', type=str, required=True,
        help='full path to the parameter yaml file')
    parser.add_argument('--mlflow-run-id', type=str, default=None,
        help='use a specific MLflow run if given')
    parser.add_argument('--mlflow-experiment', type=str, default='psg-gan',
        help='use a specific MLflow experiment if given')
    parser.add_argument('--mlflow-tracking-uri', type=str,
        default='http://localhost:5000',
        help='use a specific MLflow tracking URI if given')
    parser.add_argument('--visible-device', type=str, default='0',
        help='The cuda device to which the computation is restricted.')

    return parser


def get_config(
        param_path: str,
        mlflow_run_id: str,
        mlflow_experiment: str,
        mlflow_tracking_uri: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    # Read the yaml config file
    with open(param_path, 'r') as f:
        config = yaml.safe_load(f)

    # Construct the MLflow config
    mlflow_config = {
        'run_id': mlflow_run_id,
        'experiment': mlflow_experiment,
        'tracking_uri': mlflow_tracking_uri
    }

    return config, mlflow_config 


def run_cli() -> None:
    # https://github.com/google/grain/issues/1355
    # flags.FLAGS.mark_as_parsed()

    parser = get_parser()
    args = parser.parse_args()

    set_gpu_visibility(args.visible_device)

    config, mlflow_config = get_config(
        args.param_path,
        args.mlflow_run_id, args.mlflow_experiment, args.mlflow_tracking_uri)    
    
    train(config, mlflow_config)


if __name__ == '__main__':
    run_cli()
