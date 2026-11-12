"""Plot subject signals and annotations, along with the instantaneous heart rate from PPG."""
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sleeplab_format as slf

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from psg_gan.plot_subject import plot_subject
from scipy.interpolate import interp1d
from systole.correction import correct_peaks
from systole.detection import ppg_peaks

logger = logging.getLogger(__name__)


CH_NAME_MAP = {
    'C4M1_64Hz_hp': 'C4-M1',
    'c4m1': 'C4-M1',
    'E1M2_64Hz_hp': 'E1-M2',
    'e1m2': 'E1-M2',
    'NasalPressure_64Hz': 'Nasal\npressure',
    'naspres': 'Nasal\npressure',
    'Pleth_64Hz': 'PPG',
    'ppg': 'PPG'
}


def plot_subject_with_ihr(
        subject_dir: str,
        start_sec: float = 0.0,
        end_sec: float = -1.0,
        resample_fs: float = None,
        channels: list[str] | None = None,
        annotations: list[str] | None = None,
        ppg_label: str = 'ppg',
        ppg_axis: int = -2,
        mpl_backend: str = 'tkagg',
        x_axis_no_format: bool = False,
        ylims: list[tuple[float, float]] | None = None,
        detection_method: str = 'msptd') -> tuple[Figure, Axes]:
    assert detection_method in ('msptd', 'rolling_average')
    ppg_slf = slf.reader.read_subject(Path(subject_dir)).sample_arrays[ppg_label]
    fs = int(ppg_slf.attributes.sampling_rate)
    ppg = ppg_slf.values[int(start_sec*fs):]
    if end_sec != -1.0:
        ppg = ppg[:int((end_sec-start_sec)*fs)]

    _, peaks = ppg_peaks(
        ppg,
        fs,
        method=detection_method,
        detector_kws={'win_len': 20, 'overlap': 0.25})
    
    corrected_peaks = correct_peaks(peaks, n_iterations=2)
    fig, axs = plot_subject(
        subject_dir,
        extra_axes=1,
        matplotlib_backend=mpl_backend,
        channel_name_map=CH_NAME_MAP,
        channels=channels,
        annotations=annotations,
        start_sec=start_sec,
        end_sec=end_sec,
        resample_fs=resample_fs,
        x_axis_to_time=not x_axis_no_format)
    
    # Plot the peaks on top of ppg
    # The data has been resampled to 1000 Hz by ppg_peaks(), so transform peak indices back to 64 Hz
    corrected_peak_idx = np.rint(np.where(corrected_peaks['clean_peaks'])[0] * fs / 1000).astype(int)
    axs[ppg_axis].scatter(corrected_peak_idx, ppg[corrected_peak_idx], c='red', marker='x', s=8, linewidths=0.75)

    # Plots for corrected peaks
    ibi_corrected = np.diff(np.where(corrected_peaks['clean_peaks'])[0])
    axs[-1].scatter(
        corrected_peak_idx[1:],
        ibi_corrected,
        marker='o',
        label='IBI',
        s=10,
        color='lightgray',
        edgecolors='gray',
        zorder=2
    )
    f = interp1d(corrected_peak_idx[1:], ibi_corrected, kind='cubic', bounds_error=False, fill_value=(np.nan, np.nan))
    t = np.arange(len(ppg))
    axs[-1].plot(t, f(t), label='Interpolated', linewidth=0.75, color='#4c72b0', alpha=0.4, zorder=1)

    axs[-1].text(-0.12, 0.5, 'IBI (ms)', transform=axs[-1].transAxes,
            horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')
    
    # Change ylims if given
    if ylims is not None:
        for i, ylim in enumerate(ylims):
            axs[i].set_ylim(ylim)

    return fig, axs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject-dir', required=True)
    parser.add_argument('--start-sec', type=float, default=0.0)
    parser.add_argument('--end-sec', type=float, default=-1.0)
    parser.add_argument('-f', '--resample-fs', type=float, default=None)
    parser.add_argument('-b', '--mpl-backend', default='tkagg')
    parser.add_argument('-c', '--channels', nargs='*', default=None)
    parser.add_argument('-a', '--annotations', nargs='*', default=None),
    parser.add_argument('--ppg-label', default='ppg')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--x-axis-no-format', action='store_true')
    parser.add_argument('--ylims', type=json.loads, default=None,
        help='Optional ymin and ymax for each of the channels')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())
    save_path = args.pop('save_path')

    fig, axs = plot_subject_with_ihr(**args)
    if save_path is not None:
        logger.info(f'Saving figure to {save_path}')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if args['mpl_backend'] == 'tkagg':
        plt.show(block=True)
        
