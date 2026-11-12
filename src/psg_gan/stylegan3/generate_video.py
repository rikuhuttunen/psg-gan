"""Create a video of latent space interpolation."""
import argparse
import logging
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from datetime import timedelta
from orbax.checkpoint import v1 as ocp
from psg_gan.stylegan3 import training
from tqdm import tqdm

logger = logging.getLogger(__name__)


SIGNAL_NAME_MAP = {
    'c4m1': 'C4-M1',
    'e1m2': 'E1-M2',
    'naspres': 'Nasal\npressure',
    'ppg': 'PPG'
}


def generate_video(
        ckpt, kf_seeds, n_wframes, interval,
        zoom_start_idx=64*60*10,
        zoom_end_idx=64*60*12,
        interp_wraps=2,
        kf_pause_sec=2,
        save_path=None,
        figsize=(16, 9),
        dpi=75,
        ylim=None,
        fs=64):

    def _update(frame):
        for i, (line, zoom_line) in enumerate(zip(lines, zoom_lines)):
            line[0].set_ydata(x_gs[frame][i, :])
            zoom_line[0].set_ydata(x_gs[frame][i, zoom_start_idx:zoom_end_idx])

        seed_idx = frame // (n_wframes + pause_frames - 1)
        if frame % (n_wframes + pause_frames - 1) == 0:
            fig.suptitle(f'Latent z seed = {kf_seeds[seed_idx]}')
        elif frame % (n_wframes + pause_frames - 1) == pause_frames:
            fig.suptitle(f'Interpolating to z with seed {kf_seeds[seed_idx]}')

    logger.info('Generating zs and ws...')
    zs = np.stack([np.random.RandomState(_seed).randn(ckpt['model_config']['z_dim']) for _seed in kf_seeds])
    ws = ckpt['model'].mapping(zs)

    logger.info('Setting up ws interpolation...')
    n_keyframes = len(kf_seeds)
    x = np.arange(-n_keyframes * interp_wraps, n_keyframes * (interp_wraps + 1))
    y = np.tile(ws, [interp_wraps * 2 + 1, 1, 1])
    print(f'zs shape = {zs.shape}, ws shape = {ws.shape}, x shape = {x.shape}, y shape = {y.shape}')
    w_interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)

    x_gs = []
    logger.info('Generating samples...')
    pause_frames = (1000 // interval) * kf_pause_sec
    for i in tqdm(range((n_keyframes - 1) * n_wframes + 1)):
        w = w_interp(i / n_wframes)
        x_g = ckpt['model'].synthesis(w[np.newaxis, ...])
        x_gs.append(x_g[0])

        if i % n_wframes == 0:
            # Show the keyframe for approx `kf_pause_sec` seconds
            for _ in range(pause_frames - 1):
                x_gs.append(x_g[0])

    logger.info('Rendering video...')
    signal_names = ckpt['model_config']['input_names']
    nchannels = len(signal_names)
    fig = plt.figure(figsize=figsize, layout='constrained', dpi=dpi)
    full_fig, zoom_fig = fig.subfigures(2, 1, wspace=0.1)
    axs = full_fig.subplots(nchannels, 1, sharex=True)
    zoom_axs = zoom_fig.subplots(nchannels, 1, sharex=True)
    lines = []
    zoom_lines = []
    signal_names = ckpt['model_config']['input_names']
    for i, (ax, zoom_ax) in enumerate(zip(axs, zoom_axs)):
        sig_name = SIGNAL_NAME_MAP.get(signal_names[i]) or signal_names[i]
        # Plot the whole signal
        s = x_gs[0][i, :]
        lines.append(ax.plot(s))
        if ylim is None:
            ymin, ymax = ax.get_ylim()
        else:
            ymin, ymax = ylim
        ax.set_ylim(ymin, ymax)
        ax.axvspan(zoom_start_idx, zoom_end_idx, ymin=0.0, ymax=1.0, facecolor='tab:green', alpha=0.2)
        ax.text(-0.05, 0.5, sig_name, transform=ax.transAxes, fontsize=12, fontweight='bold', ha='center', va='center')
        ax.xaxis.set_ticks(np.arange(0, 30*60*fs + 1, 5*60*fs))
        ax.xaxis.set_major_formatter(lambda x, pos: str(timedelta(seconds=float(x)//fs)))

        zoom_ax.text(-0.05, 0.5, sig_name, transform=zoom_ax.transAxes, fontsize=12, fontweight='bold', ha='center', va='center')
        zoom_ax.set_ylim(ymin, ymax)
        zoom_ax.axvspan(zoom_start_idx, zoom_end_idx, ymin=0.0, ymax=1.0, facecolor='tab:green', alpha=0.2)
        zoom_s = s[zoom_start_idx:zoom_end_idx]
        zoom_ax.xaxis.set_ticks(np.arange(zoom_start_idx, zoom_end_idx + 1, 30*fs))
        zoom_ax.xaxis.set_major_formatter(lambda x, pos: str(timedelta(seconds=float(x)//fs)))
        zoom_lines.append(zoom_ax.plot(np.arange(zoom_start_idx, zoom_end_idx), zoom_s, color='tab:green'))

    ani = animation.FuncAnimation(fig=fig, func=_update, frames=len(x_gs), interval=interval)
    if save_path is None:
        plt.show(block=True)
    else:
        logger.info(f'Saving video to {save_path}')
        ani.save(filename=save_path, writer='ffmpeg')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt-dir', required=True,
        help='The generator checkpoint location')
    parser.add_argument('--kf-seeds', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='The seeds for generated keyframes')
    parser.add_argument('--kf-pause-sec', type=int, default=2,
        help='Seconds to pause before interpolating to next seed')
    parser.add_argument('--n-wframes', type=int, default=100,  # 8 seconds with fps 12.5
        help='Number of interpolation points between keyframes')
    parser.add_argument('--interval', type=int, default=80,  # fps 12.5
        help='Interval between consecutive frames in milliseconds')
    parser.add_argument('--save-path', default=None,
        help='Video save filepath')
    parser.add_argument('--dpi', type=int, default=75,
        help='DPI for the matplotlib figure')
    parser.add_argument('--zoom-start-idx', type=int, default=64*60*5,
        help='Start index for segment that is plotted')
    parser.add_argument('--zoom-end-idx', type=int, default=64*60*7,
        help='End index for segment that is plotted')
    parser.add_argument('--linewidth', type=float, default=0.75,
        help='Linewidth for the plots')
    parser.add_argument('--ylim', type=float, nargs=2, default=None,
        help='Optional ymin and ymax for the plots')

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_arguments())
    ckpt_dir = args.pop('ckpt_dir')
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        ckpt = training.restore_checkpoint(ckptr, restore_ema=True)

    matplotlib.rcParams['lines.linewidth'] = args.pop('linewidth')
    generate_video(ckpt=ckpt, **args)
    logger.info('Done.')
