"""Pulse rate variability related analyses on real vs fake data."""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sleeplab_format as slf

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from scipy.interpolate import interp1d
from sleeplab_format.extractor.preprocess import cheby2_filtfilt
from systole.correction import correct_peaks
from systole.detection import ppg_peaks


logger = logging.getLogger(__name__)


def plot_event_ihr(
        real_series: slf.models.Series,
        fake_series: slf.models.Series,
        real_annotation_key: str = 'profusion_aasmevents',
        fake_annotation_key: str =  'utime_aasmevents',
        real_ppg_label: str = 'Pleth_64Hz',
        fake_ppg_label: str = 'ppg',
        post_duration: float = 15.0,
        post_fs: float = 4.0,
        min_dur_between: float = 15.0,
        dur_points: int = 100,
        min_event_lenght: float = 10.0,
        max_event_length: float = np.inf,
        event_names: list[str] = ['HYPOPNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED', 'APNEA'],
        all_event_names: list[str] = ['HYPOPNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED', 'APNEA'],
        plot_ylim: tuple[int, int] = (800, 1000),
        axs: tuple[Axes, Axes] | None = None,
        save_dir: Path | None = None,
        cache_path: Path | None = None,
        save_fname: str = 'event_ihr.pdf',
        filter_ppg: bool = False) -> tuple[Figure, Axes]:
    def _compute_ibis(series, annotation_key, ppg_label):
        ibis = {'dur': [], 'post': []}
        # Go through each subject
        nsubjects = len(series.subjects)
        discard_counts = {
            'duration': 0,
            'too_close': 0,
            'too_few_points_after_last': 0,
            'ppg_peaks_error': 0,
            'correct_peaks_error': 0,
            'interp1d_error': 0}
        for idx, subj in enumerate(series.subjects.values()):
            if (idx+1) % 50 == 0:
                logger.info(f'Computing ibis for subject {idx+1}/{nsubjects}')
            ppg_sarr = subj.sample_arrays[ppg_label]
            ppg = ppg_sarr.values
            fs = ppg_sarr.attributes.sampling_rate

            # Optionally filter the PPG to 0.5 - 8.0 Hz
            if filter_ppg:
                ppg = cheby2_filtfilt(ppg.astype(np.float64), fs, 8.0, btype='lowpass').astype(np.float32)
                ppg = cheby2_filtfilt(ppg.astype(np.float64), fs, 0.5, btype='highpass').astype(np.float32)

            # Include only events that will be used to filter min_dur_between
            annotations = [a
                for a in subj.annotations[annotation_key].annotations
                if a.name in all_event_names
            ]
            # Go through each event
            for i, a in enumerate(annotations):
                if a.name not in event_names:
                    continue
                if a.duration < min_event_lenght or a.duration > max_event_length:
                    discard_counts['duration'] += 1
                    continue
                if i == len(annotations) - 1:
                    # Check that there's enough signal after the last event
                    if len(ppg) / fs - (a.start_sec + a.duration) < post_duration:
                        discard_counts['too_few_points_after_last'] += 1
                        continue
                elif annotations[i+1].start_sec - (a.start_sec + a.duration) < min_dur_between:
                    # Check that the next event does not start within min_dur_between
                    discard_counts['too_close'] += 1
                    continue
                
                # Compute peaks and IBIs during and after the events with 5s margins
                # to avoid artefacts when interpolating on the borders
                margin = int(5 * fs)
                start_idx = int(a.start_sec * fs)
                end_idx = int((a.start_sec + a.duration) * fs)
                post_end_idx = int(end_idx + post_duration * fs)

                try:
                    _, peaks = ppg_peaks(
                        ppg[start_idx-margin:post_end_idx+margin],
                        fs,
                        method='msptd',
                        detector_kws={'win_len': 20, 'overlap': 0.25}
                    )
                except ValueError as e:
                    print(repr(e))
                    logger.info(f'in subject {subj.metadata.subject_id} annotation {a}')
                    discard_counts['ppg_peaks_error'] += 1
                    continue

                try:
                    corrected = correct_peaks(peaks, n_iterations=2, verbose=False)
                except IndexError as e:
                    print(repr(e))
                    logger.info(f'in subject {subj.metadata.subject_id} annotation {a}')
                    discard_counts['correct_peaks_error'] += 1
                    continue

                # Peak indices in milliseconds from start of the window
                peak_idx = np.where(corrected['clean_peaks'])[0]
                ibi = np.diff(peak_idx)

                try:
                    # Create an interpolation function to sample evenly from the IBI series
                    f = interp1d(peak_idx[1:], ibi, kind='cubic', bounds_error=False, fill_value=(np.nan, np.nan))
                except ValueError as e:
                    logger.info(f'in subject {subj.metadata.subject_id} annotation {a}')
                    print(repr(e))
                    discard_counts['interp1d_error'] += 1
                    continue

                # Analyze IBIs during event. Note that the resolution of interpolation is millisecs
                # Start and end time in ms from start of the window
                start_idx_ms = int(5 * 1000)
                end_idx_ms = int(start_idx_ms + a.duration * 1000)

                # Evenly sample points during the event
                t_dur = np.linspace(start_idx_ms, end_idx_ms, dur_points)
                dur_ibi = f(t_dur)
                ibis['dur'].append(dur_ibi)

                # Analyze IBIs after event
                # Post period end time in ms from start of the window
                post_end_idx_ms = int(end_idx_ms + post_duration * 1000)
                post_points = int(post_fs * post_duration)
                t_post = np.linspace(end_idx_ms, post_end_idx_ms, post_points)
                post_ibi = f(t_post)
                ibis['post'].append(post_ibi)

        return ibis, discard_counts

    if cache_path is None:
        real_ibis, real_discard_counts = _compute_ibis(real_series, real_annotation_key, real_ppg_label)
        fake_ibis, fake_discard_counts = _compute_ibis(fake_series, fake_annotation_key, fake_ppg_label)
    else:
        try:
            logger.info(f'Loading IBIs from {cache_path}')
            with open(cache_path, 'rb') as cache_file:
                cached_data = pickle.load(cache_file)

            real_ibis = cached_data['real_ibis']
            real_discard_counts = cached_data['real_discard_counts']
            fake_ibis = cached_data['fake_ibis']
            fake_discard_counts = cached_data['fake_discard_counts']
        except FileNotFoundError:
            logger.info(f'Computing IBIs since cache file not found {cache_path}')
            real_ibis, real_discard_counts = _compute_ibis(real_series, real_annotation_key, real_ppg_label)
            fake_ibis, fake_discard_counts = _compute_ibis(fake_series, fake_annotation_key, fake_ppg_label)
            logger.info(f'Saving IBIs to {cache_path}')
            with open(cache_path, 'wb') as cache_file:
                pickle.dump({
                    'real_ibis': real_ibis,
                    'real_discard_counts': real_discard_counts,
                    'fake_ibis': fake_ibis,
                    'fake_discard_counts': fake_discard_counts}, cache_file)

    ####################
    # Plot the results #
    ####################
    logger.info('Plotting the results...')
    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), layout='constrained')
    
    # Plot IHR during events
    dur_t = np.linspace(0, 1, 100)
    axs[0].plot(dur_t, np.nanquantile(real_ibis['dur'], 0.5, axis=0), label=f'Real, $\it n$={len(real_ibis["dur"])}',
                linestyle='dashed', linewidth=2.5)
    axs[0].plot(dur_t, np.nanquantile(fake_ibis['dur'], 0.5, axis=0), label=f'Synthetic, $\it n$={len(fake_ibis["dur"])}',
                linewidth=2.5)
    axs[0].set_ylim(plot_ylim)
    axs[0].grid()
    if fig is not None:
        axs[0].set_title('During event', fontsize=15, fontweight='bold')
        axs[0].set_ylabel('Median IBI (ms)', fontsize=13, fontweight='bold')
        axs[0].set_xlabel('Normalized event time', fontsize=13, fontweight='bold')
    
    # Plot IHR after events
    post_t = np.linspace(0, post_duration, int(post_duration*post_fs))
    axs[1].plot(post_t, np.nanquantile(real_ibis['post'], 0.5, axis=0), label=f'Real, $\it n$={len(real_ibis["post"])}',
                linestyle='dashed', linewidth=2.5)
    axs[1].plot(post_t, np.nanquantile(fake_ibis['post'], 0.5, axis=0), label=f'Synthetic, $\it n$={len(fake_ibis["post"])}',
                linewidth=2.5)
    axs[1].set_ylim(plot_ylim)
    axs[1].legend(loc='upper right', fontsize=13)
    axs[1].grid()
    if fig is not None:
        axs[1].set_title('After event', fontsize=15, fontweight='bold')
        axs[1].set_xlabel('Time after event (s)', fontsize=13, fontweight='bold')

    if fig is not None:
        if save_dir is not None:
            fig.savefig(save_dir / save_fname, dpi=300)

    return fig, axs, real_ibis, fake_ibis, real_discard_counts, fake_discard_counts


def events_to_df(
        series: slf.models.Series,
        annotation_key: str = 'utime_aasmevents') -> pd.DataFrame:
    """Transform the SLF annotations to pandas DataFrame."""
    annotation_list = [{'sid': s.metadata.subject_id, **dict(a)}
        for s in series.subjects.values()
        for a in s.annotations[annotation_key].annotations
    ]
    annotation_df = pd.DataFrame(annotation_list)
    
    return annotation_df


def plot_ap_hyp_stats(
        real_series: slf.models.Series,
        fake_series: slf.models.Series,
        real_annotation_key: str = 'profusion_aasmevents',
        fake_annotation_key: str = 'utime_aasmevents',
        save_dir: Path | None = None) -> None:
    """Modified from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html"""
    ###############
    # Plot counts #
    ###############
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), layout='constrained')
    real_df = events_to_df(real_series, real_annotation_key)
    fake_df = events_to_df(fake_series, fake_annotation_key)

    # Rename apnea subtypes since they were not used for training
    apnea_subtypes = ['APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED']
    real_df.loc[real_df['name'].isin(apnea_subtypes), 'name'] = 'APNEA'

    rename_map = {'APNEA': 'Apneas', 'HYPOPNEA': 'Hypopneas'}
    real_df = real_df[real_df['name'].isin(list(rename_map.keys()))]
    real_counts = real_df.value_counts('name').rename(rename_map)
    fake_counts = fake_df.value_counts('name').rename(rename_map)

    bar_width = 0.4
    cnt_labels = ['Apneas', 'Hypopneas']
    x = np.arange(len(cnt_labels))
    ds_labels = ['Real', 'Synthetic']
    pastel_colors = sns.color_palette('pastel').as_hex()[:2]

    for i, counts in enumerate([real_counts, fake_counts]):
        offset = i * bar_width
        rects = axs[0].bar(x + offset, counts[cnt_labels].values, bar_width, label=ds_labels[i], color=pastel_colors[i])
        axs[0].bar_label(rects, padding=3, fontsize=12)

    axs[0].set_ylabel('Count', fontsize=12)#, fontweight='bold')
    axs[0].set_xticks(x + bar_width / 2, cnt_labels, fontsize=12)#, fontweight='bold')
    axs[0].legend(loc='upper left', ncols=2, fontsize=12)
    axs[0].set_ylim(0, 1.45 * max([real_counts.Hypopneas, fake_counts.Hypopneas]))  # Make room for legend

    ##################
    # Plot durations #
    ##################
    real_df['src'] = '1_real'
    fake_df['src'] = '2_fake'
    concat_df = pd.concat([real_df, fake_df])
    positions = [1.0, 1.6, 2.45, 3.05]
    concat_df[['src', 'name', 'duration']].boxplot(
        by=['name', 'src'],
        ax=axs[1],
        showfliers=False,
        patch_artist=True,
        positions=positions,
        widths=0.5)
    
    colors = pastel_colors * 2
    for patch, color in zip(axs[1].patches, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    
    axs[1].set_xticklabels(['Real', 'Synthetic', 'Real', 'Synthetic'], fontsize=12)#, fontweight='bold')
    axs[1].text(0.25, 1.02, 'Apneas', horizontalalignment='center', transform=axs[1].transAxes, fontsize=12)#, fontweight='bold')
    axs[1].text(0.75, 1.02, 'Hypopneas', horizontalalignment='center', transform=axs[1].transAxes, fontsize=12)#, fontweight='bold')
    axs[1].set_title('')
    fig.suptitle('')
    axs[1].set_ylabel('Duration (s)', fontsize=12)#, fontweight='bold')
    axs[1].set_xlabel('')
    axs[1].grid(axis='x')

    if save_dir is not None:
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / 'apnea_hypopnea_stats.pdf', dpi=300)
        concat_df[['src', 'name', 'duration']].groupby(['name', 'src']).describe().to_html(save_dir / 'apnea_hypopnea_durs.html', float_format='%.2f')

    return fig, axs
        

def run_analyses(
        save_dir: str,
        real_ds_dir: str,
        fake_ds_dir: str,
        real_series_name: str,
        fake_series_name: str,
        real_annotation_key: str,
        fake_annotation_key: str,
        cache_dir: str | None = None,
        filter_ppg: bool = False) -> None:
    """Run all analyses related to PRV and obstructions."""
    def _get_cache_path(cache_dir, prefix):
        if cache_dir is None:
            return None
        else:
            return Path(cache_dir) / f'{prefix}_ibis.pickle'

    logger.info('Reading real dataset...')
    real_ds = slf.reader.read_dataset(Path(real_ds_dir))
    if real_series_name == 'all':
        all_subjects = {}
        for sname in real_ds.series:
            all_subjects.update(real_ds.series[sname].subjects)
        real_series = slf.models.Series(name='all', subjects=all_subjects)
    else:
        real_series = real_ds.series[real_series_name]
    logger.info('Reading fake dataset...')
    fake_ds = slf.reader.read_dataset(Path(fake_ds_dir))
    fake_series = fake_ds.series[fake_series_name]

    logger.info('Plotting apnea and hypopnea counts and durations...')
    plot_ap_hyp_stats(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir))

    # Fig and axes for the paper
    fig, axs = plt.subplots(3, 4, figsize=(13, 6), sharex='col', sharey=True, layout='constrained')

    ####################################
    # PRV analysis on all events > 10s #
    ####################################
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='event_ihr.pdf', plot_ylim=(750, 1050))
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='apnea_ihr.pdf', plot_ylim=(750, 1050),
    #     event_names=['APNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED'])
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='hypopnea_ihr.pdf', plot_ylim=(750, 1050),
    #     event_names=['HYPOPNEA'])

    ######################################
    # PRV analysis on 10s < events < 20s #
    ######################################
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='event_ihr_10-20s.pdf', plot_ylim=(750, 1050),
    #     min_event_lenght=10.0, max_event_length=20.0)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='apnea_ihr_10-20s.pdf', plot_ylim=(800, 1100),
        event_names=['APNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED'],
        axs=axs[0, 2:], min_event_lenght=10.0, max_event_length=20.0,
        cache_path=_get_cache_path(cache_dir, 'apnea_10-20s'), filter_ppg=filter_ppg)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='hypopnea_ihr_10-20s.pdf', plot_ylim=(800, 1100),
        event_names=['HYPOPNEA'],
        axs=axs[0, :2], min_event_lenght=10.0, max_event_length=20.0,
        cache_path=_get_cache_path(cache_dir, 'hypopnea_10-20s'), filter_ppg=filter_ppg)


    ######################################
    # PRV analysis on 20s < events < 30s #
    ######################################
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='event_ihr_20-30s.pdf', plot_ylim=(750, 1050),
    #     min_event_lenght=20.0, max_event_length=30.0)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='apnea_ihr_20-30s.pdf', plot_ylim=(800, 1100),
        event_names=['APNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED'],
        axs=axs[1, 2:], min_event_lenght=20.0, max_event_length=30.0,
        cache_path=_get_cache_path(cache_dir, 'apnea_20-30s'), filter_ppg=filter_ppg)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='hypopnea_ihr_20-30s.pdf', plot_ylim=(800, 1100),
        event_names=['HYPOPNEA'],
        axs=axs[1, :2], min_event_lenght=20.0, max_event_length=30.0,
        cache_path=_get_cache_path(cache_dir, 'hypopnea_20-30s'), filter_ppg=filter_ppg)


    ################################
    # PRV analysis on events > 30s #
    ################################
    # _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
    #     save_dir=Path(save_dir), save_fname='event_ihr_30s.pdf', plot_ylim=(750, 1050),
    #     min_event_lenght=30.0)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='apnea_ihr_30s.pdf', plot_ylim=(800, 1100),
        event_names=['APNEA', 'APNEA_OBSTRUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED'],
        axs=axs[2, 2:], min_event_lenght=30.0,
        cache_path=_get_cache_path(cache_dir, 'apnea_30s'), filter_ppg=filter_ppg)
    _ = plot_event_ihr(real_series, fake_series, real_annotation_key, fake_annotation_key,
        save_dir=Path(save_dir), save_fname='hypopnea_ihr_30s.pdf', plot_ylim=(800, 1100),
        event_names=['HYPOPNEA'],
        axs=axs[2, :2], min_event_lenght=30.0,
        cache_path=_get_cache_path(cache_dir, 'hypopnea_30s'), filter_ppg=filter_ppg)

    axs[0, 0].set_title('During event', fontsize=15)#, fontweight='bold')
    axs[0, 1].set_title('After event', fontsize=15)#, fontweight='bold')
    axs[0, 2].set_title('During event', fontsize=15)#, fontweight='bold')
    axs[0, 3].set_title('After event', fontsize=15)#, fontweight='bold')
    axs[0, 0].set_ylabel('Median IBI (ms)', fontsize=13)#, fontweight='bold')
    axs[1, 0].set_ylabel('Median IBI (ms)', fontsize=13)#, fontweight='bold')
    axs[2, 0].set_ylabel('Median IBI (ms)', fontsize=13)#, fontweight='bold')
    axs[-1, 0].set_xlabel('Normalized event time', fontsize=13)#, fontweight='bold')
    axs[-1, 1].set_xlabel('Time after event (s)', fontsize=13)#, fontweight='bold')
    axs[-1, 2].set_xlabel('Normalized event time', fontsize=13)#, fontweight='bold')
    axs[-1, 3].set_xlabel('Time after event (s)', fontsize=13)#, fontweight='bold')

    axs[0, 0].text(-0.3, 0.5, '10 - 20 s', transform=axs[0, 0].transAxes, horizontalalignment='right',
                   fontsize=15)#, fontweight='bold')
    axs[1, 0].text(-0.3, 0.5, '20 - 30 s', transform=axs[1, 0].transAxes, horizontalalignment='right',
                   fontsize=15)#, fontweight='bold')
    axs[2, 0].text(-0.3, 0.5, '> 30 s', transform=axs[2, 0].transAxes, horizontalalignment='right',
                   fontsize=15)#, fontweight='bold')
    fig.text(0.35, 1.03, 'Hypopneas', transform=fig.transFigure, horizontalalignment='center', fontsize=15, fontweight='bold')
    fig.text(0.78, 1.03, 'Apneas', transform=fig.transFigure, horizontalalignment='center', fontsize=15, fontweight='bold')

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / 'apnea_hypopnea_ihrs.pdf', dpi=300, bbox_inches='tight')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--real-ds-dir', required=True)
    parser.add_argument('--fake-ds-dir', required=True)
    parser.add_argument('--real-series-name', default='all')
    parser.add_argument('--fake-series-name', default='stylegan3')
    parser.add_argument('--real-annotation-key', default='profusion_aasmevents')
    parser.add_argument('--fake-annotation-key', default='utime_aasmevents')
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--filter-ppg', action='store_true',
        help='If given, filter the PPG signals to [0.5, 8.0] Hz before peak detection')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_analyses(**vars(args))
