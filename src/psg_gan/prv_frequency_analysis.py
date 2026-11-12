"""Frequency band analyses for PPG-based pulse rate variability between real and synthetic data."""
import argparse
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sleeplab_format as slf

from pathlib import Path
from scipy import stats
from sleeplab_format.extractor.preprocess import cheby2_filtfilt
from systole.correction import correct_peaks
from systole.detection import ppg_peaks
from systole.hrv import frequency_domain
from tqdm import tqdm


logger = logging.getLogger(__name__)


def detect_ppg_peaks(
        ds_dir: Path,
        ppg_key: str,
        hypnogram_key: str,
        peaks_dir: Path,
        win_epochs: int = 10,
        min_ratio: float = 0.8,
        filter_ppg: bool = False) -> None:
    """Detect PPG peaks for all subjects in a SLF dataset and save to `peaks_dir`."""
    logger.info(f'Reading dataset from {ds_dir}...')
    ds = slf.reader.read_dataset(ds_dir)
    for sname in ('Wake', 'NREM', 'REM'):
        (peaks_dir / ds.name / sname).mkdir(exist_ok=True)
    for series in ds.series.values():
        # Save all series' peaks to the same folder
        logger.info(f'Finding PPG peaks for series {series.name}...')
        for subj in tqdm(series.subjects.values()):
            ppg_sarr = subj.sample_arrays[ppg_key]
            ppg = ppg_sarr.values
            fs = ppg_sarr.attributes.sampling_rate
            hg = subj.annotations[hypnogram_key].annotations

            # Optionally filter PPG to [0.5. 8.0] Hz
            if filter_ppg:
                ppg = cheby2_filtfilt(ppg.astype(np.float64), fs, 8.0, btype='lowpass').astype(np.float32)
                ppg = cheby2_filtfilt(ppg.astype(np.float64), fs, 0.5, btype='highpass').astype(np.float32)

            # Detect peaks in 5-min sliding windows separately for Wake, NREM, REM
            for stage in ('Wake', 'NREM', 'REM'):
                start = 0
                min_epochs = math.ceil(win_epochs * min_ratio)
                while (start + win_epochs) < len(hg):
                    hg_win = hg[start:start+win_epochs]
                    if stage == 'Wake':
                        stage_sum = sum([s.name == 'W' for s in hg_win])
                    elif stage == 'NREM':
                        stage_sum = sum([s.name in ('N1', 'N2', 'N3') for s in hg_win])
                    else:
                        stage_sum = sum([s.name == 'R' for s in hg_win])

                    if stage_sum >= min_epochs:
                        start += win_epochs
                        start_idx = int(hg_win[0].start_sec * fs)
                        end_idx = int((hg_win[-1].start_sec + hg_win[-1].duration) * fs)
                        try:
                            ppg_win = ppg[start_idx:end_idx]
                            _, peaks = ppg_peaks(
                                ppg_win,
                                fs,
                                verbose=True,
                                method='msptd',
                                detector_kws={'win_len': 20, 'overlap': 0.25}
                            )
                        except ValueError as e:
                            print(repr(e))
                            logger.info(f'in subject {subj.metadata.subject_id}')
                            continue
                        except IndexError as e:
                            print(repr(e))
                            logger.info(f'in subject {subj.metadata.subject_id}')
                            continue

                        try:
                            corrected = correct_peaks(peaks, n_iterations=2, verbose=False)
                            # These will be the peak locations from beginning in milliseconds since ppg_peaks resamples signal to 1000 Hz
                            clean_peaks = np.where(corrected['clean_peaks'])[0]
                            with open(peaks_dir / ds.name / stage / f'{subj.metadata.subject_id}_{start}.npy', 'wb') as peaks_file:
                                np.save(peaks_file, clean_peaks)
                        except IndexError as e:
                            print(repr(e))
                            logger.info(f'in subject {subj.metadata.subject_id}')
                            continue
                    else:
                        start += 1

    return None


def rm_outliers(arr, iqr_factor=1.5):
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    return arr[np.logical_and(arr > lower_bound, arr < upper_bound)]


def compute_features(
        peaks_dir: Path,
        window_length: float = 5 * 60.0) -> None:
    """Compute frequency features for all peak series in `peaks_dir`."""
    freq_feats = {'LF': [], 'HF': [], 'LF/HF': []}
    for peaks_fpath in peaks_dir.iterdir():
        clean_peaks = np.load(peaks_fpath)

        if len(clean_peaks) < window_length / 2:
            logger.info(f'skipping window with only {len(clean_peaks)} peaks...')
            continue
        _feats = frequency_domain(clean_peaks, input_type='peaks_idx')
        _feats = pd.pivot_table(_feats, values='Values', columns='Metric')
        freq_feats['LF'].append(_feats['lf_power_per'].Values)
        freq_feats['HF'].append(_feats['hf_power_per'].Values)
        freq_feats['LF/HF'].append(_feats['lf_hf_ratio'].Values)

    return {k: np.array(v) for k, v in freq_feats.items()}


def run_analyses(
        peaks_dir: Path,
        fig_savedir: Path,
        real_ds_dir: Path,
        fake_ds_dir: Path,
        real_ppg_key: str = 'Pleth_64Hz',
        fake_ppg_key: str = 'Pleth_64Hz',
        real_hg_key: str = 'fortunate-foal-574_hypnogram',
        fake_hg_key: str = 'fortunate-foal-574_hypnogram',
        window_length: float = 5*60,
        filter_ppg: bool = False) -> None:
    peaks_dir.mkdir(exist_ok=True, parents=True)
    if not (peaks_dir / real_ds_dir.name).is_dir():
        (peaks_dir / real_ds_dir.name).mkdir()
        logger.info('Detecting real peaks...')
        detect_ppg_peaks(real_ds_dir, real_ppg_key, real_hg_key, peaks_dir, filter_ppg=filter_ppg)
    if not (peaks_dir / fake_ds_dir.name).is_dir():
        (peaks_dir / fake_ds_dir.name).mkdir()
        logger.info('Detecting fake peaks...')
        detect_ppg_peaks(fake_ds_dir, fake_ppg_key, fake_hg_key, peaks_dir, filter_ppg=filter_ppg)

    stages = ('Wake', 'NREM', 'REM')
    real_feats = {s: {} for s in stages}
    fake_feats = {s: {} for s in stages}
    for stage in stages:
        logger.info(f'Computing real PRV frequency features for {stage}...')
        real_feats[stage] = compute_features(
            peaks_dir / real_ds_dir.name / stage,
            window_length=window_length
        )
        logger.info(f'Computing fake PRV frequency features for {stage}...')
        fake_feats[stage] = compute_features(
            peaks_dir / fake_ds_dir.name / stage,
            window_length=window_length
        )

    bands = ['LF', 'HF', 'LF/HF']
    logger.info('Running statistical tests...')
    ks_tests = {s: {} for s in stages}
    es_tests = {s: {} for s in stages}
    cvm_tests = {s: {} for s in stages}
    for stage in stages:
        for band in bands:
            kstest = stats.ks_2samp(rm_outliers(real_feats[stage][band]), rm_outliers(fake_feats[stage][band]))
            ks_tests[stage][band] = {'pvalue': kstest.pvalue, 'statistic': kstest.statistic}
            estest = stats.epps_singleton_2samp(rm_outliers(real_feats[stage][band]), rm_outliers(fake_feats[stage][band]))
            es_tests[stage][band] = {'pvalue': estest.pvalue, 'statistic': estest.statistic}
            cvmtest = stats.cramervonmises_2samp(rm_outliers(real_feats[stage][band]), rm_outliers(fake_feats[stage][band]))
            cvm_tests[stage][band] = {'pvalue': cvmtest.pvalue, 'statistic': cvmtest.statistic}

    logger.info('Plotting violin plots with seaborn...')
    # Transform to wide-form dataframe for seaborn plotting
    feat_df = pd.concat([
        pd.DataFrame([{'Stage': s, 'Band': b, 'Value': v, 'Dataset': 'Real'}
                      for s, bs in real_feats.items() for b, vs in bs.items() for v in rm_outliers(vs)]),
        pd.DataFrame([{'Stage': s, 'Band': b, 'Value': v, 'Dataset': 'Synthetic'}
                      for s, bs in fake_feats.items() for b, vs in bs.items() for v in rm_outliers(vs)])
    ])
    sns_fig, sns_axs = plt.subplots(3, 1, figsize=(6, 6), layout='constrained')
    for i, band in enumerate(bands):
        sns.violinplot(
            data=feat_df[feat_df['Band'] == band], x='Stage', y='Value', hue='Dataset',
            split=True, gap=0.05, inner='quart', ax=sns_axs[i], cut=0, palette='pastel', common_norm=True,
            order=['Wake', 'REM', 'NREM']
        )
        sns_axs[i].grid(axis='y')
        sns_axs[i].set_xlabel(None)

    sns_axs[0].set_xticks([])
    sns_axs[1].set_xticks([])

    sns_axs[0].set_ylabel('LF power (%)', fontsize=11)#, fontweight='bold')
    sns_axs[1].set_ylabel('HF power (%)', fontsize=11)#, fontweight='bold')
    sns_axs[2].set_ylabel('LF/HF ratio', fontsize=11)#, fontweight='bold')

    sns.move_legend(sns_axs[0], 'lower center', bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False)
    sns_axs[1].get_legend().remove()
    sns_axs[2].get_legend().remove()

    stats_path = fig_savedir / 'prv_band_stats.parquet'
    logger.info(f'Saving statistics on the data to {stats_path}')
    stats_df = feat_df.groupby(['Stage', 'Band', 'Dataset']).describe()
    stats_df = stats_df[[('Value', '25%'), ('Value', '50%'), ('Value', '75%')]]
    stats_df.columns = stats_df.columns.set_levels(['Quantile'], level=0)
    stats_df = stats_df.sort_index(axis=0, level=(0,1,2), ascending=(True, True, False))
    stats_df.to_parquet(stats_path)
    stats_df.to_html(fig_savedir / 'prv_band_stats.html', float_format='%.2f')

    logger.info('Plotting violin plots with matplotlib...')
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey='row', layout='constrained')
    vp_width = 0.7
    vp_centers = list(range(1, len(stages) + 1))
    vp_ticklocations = [c + d for c in vp_centers for d in ((-vp_width/4), vp_width/4)]
    tick_labels = ['Real', 'Fake', 'Real', 'Fake', 'Real', 'Fake']
    for i, band in enumerate(bands):

        real_vplot = axs[i].violinplot(
            [
                rm_outliers(real_feats['Wake'][band]),
                rm_outliers(real_feats['NREM'][band]),
                rm_outliers(real_feats['REM'][band])
            ],
            positions=vp_centers,
            widths=vp_width,
            showmedians=True,
            quantiles = [[0.25, 0.75]] * len(stages),
            side='low'
        )
        fake_vplot = axs[i].violinplot(
            [
                rm_outliers(fake_feats['Wake'][band]),
                rm_outliers(fake_feats['NREM'][band]),
                rm_outliers(fake_feats['REM'][band])
            ],
            positions=vp_centers,
            widths=vp_width,
            showmedians=True,
            quantiles = [[0.25, 0.75]] * len(stages),
            side='high'
        )
        axs[i].grid(axis='y')
        for patch in real_vplot['bodies']:
            patch.set_facecolor('tab:blue')
            patch.set_alpha(0.4)
            patch.set_edgecolor('black')

        for patch in fake_vplot['bodies']:
            patch.set_facecolor('tab:orange')
            patch.set_alpha(0.4)
            patch.set_edgecolor('black')

        for vplot in (real_vplot, fake_vplot):
            vplot['cmedians'].set_color('black')
            vplot['cmedians'].set_linestyle('dashed')
            vplot['cmedians'].set_linewidth(1)
            vplot['cquantiles'].set_color('black')
            vplot['cquantiles'].set_linestyle('dashed')
            vplot['cquantiles'].set_linewidth(0.5)
            vplot['cmaxes'].set_color('black')
            vplot['cmaxes'].set_linewidth(1)
            vplot['cmins'].set_color('black')
            vplot['cmins'].set_linewidth(1)
            vplot['cbars'].set_alpha(0)

        if i == len(bands) - 1:
            axs[i].set_xticks(vp_ticklocations)
            axs[i].set_xticklabels(tick_labels, rotation=45, ha='right',
                                   rotation_mode='anchor', fontsize=10)
        else:
            axs[i].set_xticks([])

    for stage, center in zip(stages, vp_centers):
        axs[0].text(center, 1.05, stage, fontsize=12, fontweight='bold', ha='center',
                    transform=axs[0].get_xaxis_transform())

    axs[0].set_ylabel('LF power (%)', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('HF power (%)', fontsize=11, fontweight='bold')
    axs[2].set_ylabel('LF/HF ratio', fontsize=11, fontweight='bold')

    sns_fig.savefig(fig_savedir / 'prv_freq_violin_sns.pdf', dpi=600, bbox_inches='tight')
    fig.savefig(fig_savedir / 'prv_freq_violin.pdf', dpi=600, bbox_inches='tight')

    with open(Path(fig_savedir) / 'PRV_ks_test_results.json', 'w') as f:
        json.dump(ks_tests, f, indent=2)
    with open(Path(fig_savedir) / 'PRV_epps_singleton_test_results.json', 'w') as f:
        json.dump(es_tests, f, indent=2)
    with open(Path(fig_savedir) / 'PRV_cramer_vonmises_test_results.json', 'w') as f:
        json.dump(cvm_tests, f, indent=2)
    
    return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peaks-dir', type=Path, required=True)
    parser.add_argument('--fig-savedir', type=Path, required=True)
    parser.add_argument('--real-ds-dir', type=Path, required=True)
    parser.add_argument('--fake-ds-dir', type=Path, required=True)
    parser.add_argument('--real-ppg-key', default='Pleth_64Hz')
    parser.add_argument('--fake-ppg-key', default='Pleth_64Hz')
    parser.add_argument('--real-hg-key', default='fortunate-foal-574_hypnogram')
    parser.add_argument('--fake-hg-key', default='fortunate-foal-574_hypnogram')
    parser.add_argument('--window-length', type=float, default=5*60)
    parser.add_argument('--filter-ppg', action='store_true',
        help='If given, filter the PPG signals to [0.5, 8.0] Hz before peak detection')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_analyses(**vars(args))
