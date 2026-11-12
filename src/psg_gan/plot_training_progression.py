"""Plot GAN training progression."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path


def get_x_y(df, key):
    xy_df = df[df['metric'] == key][['step', 'value']]

    return xy_df['step'].values, xy_df['value'].values


def plot_training_progression(csv_path, save_dir):
    """Plot the progression of GAN losses and FUD during training.
    
    The CSV can be exported from the MLflow UI plot.
    """
    df = pd.read_csv(csv_path)
    fig, axs = plt.subplots(3, 1, sharex=True, layout='constrained', figsize=(6.4, 4.8))

    steps, values = get_x_y(df, 'fud')
    axs[0].semilogy(steps, values, label='FUD', color='tab:blue')
    axs[0].set_ylim(3.0, 100.0)
    axs[0].set_ylabel('FUD', fontsize=12)#, fontweight='bold')

    steps, values = get_x_y(df, 'g_loss')
    axs[1].semilogy(steps, values, label='Generator loss', color='tab:blue')
    axs[1].set_ylim(0.15, 10.0)
    axs[1].set_ylabel('G loss', fontsize=12)#, fontweight='bold')

    steps, values = get_x_y(df, 'd_loss')
    axs[2].semilogy(steps, values, label='Discriminator loss', color='tab:blue')
    axs[2].set_ylabel('D loss', fontsize=12)#, fontweight='bold')
    axs[2].set_xlabel('Training samples shown to the model (Million)', fontsize=11)#, fontweight='bold')
    xticks = np.linspace(0, 2, 9)
    xticklabels = [f'{t:.2f}' for t in xticks]
    axs[2].set_xticks(xticks * 1e6, xticklabels)

    for ax in axs:
        ax.grid()

    fig.savefig(Path(save_dir) / 'stylegan3_training_progression.pdf', dpi=300, bbox_inches='tight')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True)
    parser.add_argument('--save-dir', required=True)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    plot_training_progression(**vars(args))
