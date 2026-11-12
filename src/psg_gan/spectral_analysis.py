"""Assess generator quality using spectral features."""
import argparse
import copy
import jax
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
import yaml

from functools import partial
from pathlib import Path
from psg_gan.metrics.spectral import compute_psds
from scipy.integrate import simpson
from scipy import stats
from tqdm import tqdm


logger = logging.getLogger(__name__)


def compute_psds_from_slf(cfg):
    import grain
    import slf_grain
    logger.info('Reading real dataset...')

    series_names = cfg['real']['series_name']
    series_datasets = []
    for series_name in series_names:
        _cfg = copy.deepcopy(cfg['real'])
        _cfg['series_name'] = series_name
        _ds = slf_grain.dataset.from_slf(slf_grain.config.DatasetConfig.model_validate(_cfg))
        series_datasets.append(_ds)

    real_ds = grain.MapDataset.concatenate(series_datasets)

    logger.info('Reading fake dataset...')
    fake_ds = slf_grain.dataset.from_slf(slf_grain.config.DatasetConfig.model_validate(cfg['fake']))

    all_keys = set(cfg['real']['components'].keys())
    input_names = sorted(list(all_keys - set(['hypnogram'])), key=str.casefold)

    def _configure_ds(ds: grain.MapDataset):
        return (ds
            .map(lambda d: {
                'signals': np.concatenate([d[k] for k in input_names], axis=-1),
                'hypnogram': d['hypnogram']})
            .batch(1)
            .to_iter_dataset()
        )

    fs = cfg['real']['components'][input_names[0]]['fs']
    _compute_psds = jax.jit(jax.vmap(partial(
        compute_psds,
        fs=fs, epoch_length=30, window='hann', nperseg=int(fs*30/4), noverlap=None,
        nfft=None, detrend='constant', return_onesided=True, scaling='density',
        axis=-2, average='mean'
        )
    ))

    real_psds = []
    real_hgs = []
    fake_psds = []
    fake_hgs = []

    logger.info('Computing real PSDs...')
    for e in tqdm(_configure_ds(real_ds)):
        _, _psd = _compute_psds(e['signals'])
        _hg = np.asarray(e['hypnogram'][0, ...])
        real_psds.append(np.asarray(_psd[0, ...]))
        real_hgs.append(_hg)
    
    real_psds = np.concatenate(real_psds, axis=0)
    real_hgs = np.concatenate(real_hgs, axis=0)

    logger.info('Computing fake PSDs...')
    for e in tqdm(_configure_ds(fake_ds)):
        _, _psd = _compute_psds(e['signals'])
        _hg = np.asarray(e['hypnogram'][0, ...])
        fake_psds.append(np.asarray(_psd[0, ...]))
        fake_hgs.append(_hg)
    
    fake_psds = np.concatenate(fake_psds, axis=0)
    fake_hgs = np.concatenate(fake_hgs, axis=0)

    return real_psds, real_hgs, fake_psds, fake_hgs, input_names, fs


def rm_outliers(arr, iqr_factor=1.5):
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    return arr[np.logical_and(arr > lower_bound, arr < upper_bound)]


def plot_eeg_bandpowers(
        freqs,
        real_psds,  # [NF]
        fake_psds,
        real_hgs,  # [N]
        fake_hgs,
        stages,
        labels,
        fig_savedir) -> None:
    logger.info(f'Computing and plotting EEG bandpowers for {labels}...')
    logger.info(f'real_psds.shape = {real_psds.shape}, real_hgs.shape = {real_hgs.shape}')
    logger.info(f'fake_psds.shape = {fake_psds.shape}, fake_hgs.shape = {fake_hgs.shape}')

    band_idx = {
        'Delta': np.logical_and(freqs >= 0.0, freqs < 4.0),
        'Theta': np.logical_and(freqs >= 4.0, freqs < 8.0),
        'Alpha': np.logical_and(freqs >= 8.0, freqs <= 13.0),
        'Beta': np.logical_and(freqs > 13.0, freqs <= 30.0)
    }

    fig_sns, axs_sns = plt.subplots(len(band_idx), len(labels), figsize=(6*len(labels), 2.5*len(band_idx)),
                                  layout='constrained', sharey='row')

    freq_res = freqs[1] - freqs[0]
    ks_tests = {l: {s: {} for s in stages} for l in labels}
    epps_singleton_tests = {l: {s: {} for s in stages} for l in labels}
    cramer_vonmises_tests = {l: {s: {} for s in stages} for l in labels}

    # Perform analyses separately for each label, stage, and band
    for h, label in enumerate(labels):
        real_bps = {s: {} for s in stages}
        fake_bps = {s: {} for s in stages}
        for i, stage in enumerate(stages):
            for j, band in enumerate(band_idx.keys()):
                # Calculate relative band powers
                logger.info(f'Calculating {band} band powers for real {label} signals for {stage}...')
                _real_bps = np.array([
                    simpson(psd[band_idx[band]], dx=freq_res) / simpson(psd, dx=freq_res)
                    for psd in real_psds[..., h][real_hgs==i]])
                # Remove NaNs
                _real_bps = _real_bps[~np.isnan(_real_bps)]
                real_bps[stage][band] = _real_bps

                logger.info(f'Calculating {band} band powers for fake {label} signals for {stage}...')
                _fake_bps = np.array([
                    simpson(psd[band_idx[band]], dx=freq_res) / simpson(psd, dx=freq_res)
                    for psd in fake_psds[..., h][fake_hgs==i]])
                _fake_bps = _fake_bps[~np.isnan(_fake_bps)]
                fake_bps[stage][band] = _fake_bps

                # Two-sample tests for statistically significant difference
                # Normal YES:
                # - ANOVA?
                # - MANOVA?
                # - Student's t?
                # - Welch's t?
                
                # Normal NO:
                # - Kolmogorow-Smirnov
                # - Mann-whitney U?

                # Test whether the distributions are different
                logger.info(f'Running statistical tests for {label} {stage} {band}...')
                kstest = stats.ks_2samp(rm_outliers(_real_bps), rm_outliers(_fake_bps))
                ks_tests[label][stage][band] = {'pvalue': kstest.pvalue, 'statistic': kstest.statistic,
                                                'statistic_location': kstest.statistic_location,
                                                'statistic_sign': int(kstest.statistic_sign)}
                estest = stats.epps_singleton_2samp(rm_outliers(_real_bps), rm_outliers(_fake_bps))
                epps_singleton_tests[label][stage][band] = {'pvalue': estest.pvalue, 'statistic': estest.statistic}
                cvmtest = stats.cramervonmises_2samp(rm_outliers(_real_bps), rm_outliers(_fake_bps))
                cramer_vonmises_tests[label][stage][band] = {'pvalue': cvmtest.pvalue, 'statistic': cvmtest.statistic}

        logger.info('Plotting seaborn violin plots...')
        feat_df = pd.concat([
            pd.DataFrame([{'Stage': s, 'Band': b, 'Value': v * 100, 'Dataset': 'Real'}
                         for s, bs in real_bps.items() for b, vs in bs.items() for v in rm_outliers(vs)]),
            pd.DataFrame([{'Stage': s, 'Band': b, 'Value': v * 100, 'Dataset': 'Synthetic'}
                         for s, bs in fake_bps.items() for b, vs in bs.items() for v in rm_outliers(vs)])
        ])
        widths = {'Delta': 1.2, 'Theta': 1.0, 'Alpha': 1.4, 'Beta': 1.6}
        sns.set_theme(font_scale=1.5)
        for i, band in enumerate(band_idx.keys()):
            sns.violinplot(
                data=feat_df[feat_df['Band'] == band], x='Stage', y='Value', hue='Dataset', palette='pastel',
                split=True, width=widths[band], gap=0.075/widths[band], inner='quart', ax=axs_sns[i, h], cut=0,
                order=['W', 'REM', 'N1', 'N2', 'N3']
            )
            axs_sns[i, h].grid(axis='y')
            axs_sns[i, h].set_xlabel(None)
            if h == 0:
                axs_sns[i, h].set_ylabel(f'{band}', fontsize=15)
            axs_sns[i, h].tick_params(labelsize=14)

            # Modify the xlims to fit all data to the plots...
            max_width = max(widths.values())
            xlims = axs_sns[i, h].get_xlim()
            axs_sns[i, h].set_xlim(xlims[0] + max_width/2 - 0.6, xlims[1] + max_width / 2 - 0.5)

        axs_sns[0, h].set_xticks([])
        axs_sns[1, h].set_xticks([])
        axs_sns[2, h].set_xticks([])

        sns.move_legend(axs_sns[0, h], 'lower center', bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False, fontsize=15)
        axs_sns[1, h].get_legend().remove()
        axs_sns[2, h].get_legend().remove()
        axs_sns[3, h].get_legend().remove()
        axs_sns[0, h].set_title(label, fontsize=18, fontweight='bold', pad=40)

        logger.info(f'Saving statistics on the data...')
        stats_df = feat_df.groupby(['Stage', 'Band', 'Dataset']).describe()
        stats_df = stats_df[[('Value', '25%'), ('Value', '50%'), ('Value', '75%')]]
        stats_df.columns = stats_df.columns.set_levels(['Quantile'], level=0)
        stats_df = stats_df.sort_index(axis=0, level=(0,1,2), ascending=(True, True, False))
        stats_df.to_html(fig_savedir / f'{label}_band_stats.html', float_format='%.2f')

    fig_sns.supylabel('Relative power (%)', fontsize=15)
    fig_sns.savefig(Path(fig_savedir) / f'{"_".join(labels)}_bandpowers_violin_sns.pdf', dpi=600, bbox_inches='tight')
    
    with open(Path(fig_savedir) / 'EEG_ks_test_results.json', 'w') as f:
        json.dump(ks_tests, f, indent=2)
    with open(Path(fig_savedir) / 'EEG_epps_singleton_test_results.json', 'w') as f:
        json.dump(epps_singleton_tests, f, indent=2)
    with open(Path(fig_savedir) / 'EEG_cramer_vonmises_test_results.json', 'w') as f:
        json.dump(cramer_vonmises_tests, f, indent=2)

    return None


def plot_median_psds_from_slf_ds(
        cfg_path: Path,
        fig_savedir: Path,
        copy_dst: Path | None = None,
        cache_path: Path | None = None) -> None:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if copy_dst is not None:
        # Copy the datasets to another location before running analyses
        src_dir = Path(cfg['real']['ds_dir'])
        dst_dir = copy_dst / src_dir.name
        if not dst_dir.exists():
            logger.info(f'Copying {src_dir} to {dst_dir}')
            shutil.copytree(src_dir, dst_dir)
        cfg['real']['ds_dir'] = str(dst_dir)
        
        src_dir = Path(cfg['fake']['ds_dir'])
        dst_dir = copy_dst / src_dir.name
        if not dst_dir.exists():
            logger.info(f'Copying {src_dir} to {dst_dir}')
            shutil.copytree(src_dir, dst_dir)
        cfg['fake']['ds_dir'] = str(dst_dir)

    if cache_path is None:
        real_psds, real_hgs, fake_psds, fake_hgs, input_names, fs = compute_psds_from_slf(cfg)
    else:
        logger.info(f'Reading PSDs from {cache_path}')
        try:
            npzfile = np.load(cache_path)
            real_psds = npzfile['real_psds']
            real_hgs = npzfile['real_hgs']
            fake_psds = npzfile['fake_psds']
            fake_hgs = npzfile['fake_hgs']
            input_names = npzfile['input_names']
            fs = npzfile['fs']
        except FileNotFoundError:
            logger.info(f'Cache file not found, computing the PSDs...')
            real_psds, real_hgs, fake_psds, fake_hgs, input_names, fs = compute_psds_from_slf(cfg)
            with open(cache_path, 'wb') as npzfile:
                np.savez(npzfile,
                         real_psds=real_psds,
                         real_hgs=real_hgs,
                         fake_psds=fake_psds,
                         fake_hgs=fake_hgs,
                         input_names=input_names,
                         fs=fs)

    # Plot median PSDs
    logger.info('Plotting results...')
    nchs = len(input_names)

    fig_q, axs_q = plt.subplots(nchs, 2, sharex=True, sharey='row', figsize=(15, 3*nchs), layout='constrained')
    fig_m, axs_m = plt.subplots(nchs, 1, sharex=True, sharey=False, figsize=(10, 3*nchs), layout='constrained')

    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    fig_s, axs_s = plt.subplots(len(stages), nchs, sharex='col', sharey=True,
                                figsize=(3*nchs, 1.5*len(stages)), layout='constrained')
    fig_s_counts, axs_s_counts = plt.subplots(figsize=(6, 2.3), layout='constrained')
    
    # Plot sleep stage counts for real and fake
    real_s_counts = [np.sum(real_hgs==i) for i in range(len(stages))]
    fake_s_counts = [np.sum(fake_hgs==i) for i in range(len(stages))]
    bar_width = 0.45
    x = np.arange(len(stages))
    ds_labels = ['Real', 'Synthetic']
    pastel_colors = sns.color_palette('pastel').as_hex()[:2]
    for i, counts in enumerate([real_s_counts, fake_s_counts]):
        offset = i * bar_width
        _ = axs_s_counts.bar(x + offset, counts, bar_width, label=ds_labels[i], color=pastel_colors[i])
    axs_s_counts.grid(axis='y')

    axs_s_counts.set_ylabel('Count', fontsize=12)#, fontweight='bold')
    axs_s_counts.set_xticks(x + bar_width / 2, stages, fontsize=12)#, fontweight='bold')
    axs_s_counts.legend(loc='upper left', ncols=2, fontsize=12)
    axs_s_counts.set_ylim(0, 1.3 * max(real_s_counts + fake_s_counts))  # Make room for legend
    
    fig_s_counts.savefig(fig_savedir / 'stage_counts.pdf', dpi=300, bbox_inches='tight')

    freqs = np.linspace(0, fs/2, real_psds.shape[-2])
    
    plot_eeg_bandpowers(
        freqs,
        real_psds[..., 0:2],
        fake_psds[..., 0:2],
        real_hgs,
        fake_hgs,
        stages,
        labels=['C4-M1', 'E1-M2'],
        fig_savedir=fig_savedir
    )
    
    # Plot naspres and ppg for frequencies less than 8 Hz
    freqs_lt_8hz = freqs[freqs <= 8.0]
    real_qs = np.quantile(real_psds, q=[0.25, 0.5, 0.75], axis=0)  # [eC]
    fake_qs = np.quantile(fake_psds, q=[0.25, 0.5, 0.75], axis=0)

    real_qs_stage = {}
    fake_qs_stage = {}

    for i, stage in enumerate(stages):
        real_qs_stage[stage] = np.quantile(real_psds[real_hgs==i], q=[0.25, 0.5, 0.75], axis=0)
        fake_qs_stage[stage] = np.quantile(fake_psds[fake_hgs==i], q=[0.25, 0.5, 0.75], axis=0)

    for ch in range(nchs):
        axs_q[ch, 0].plot(freqs, 10*np.log10(real_qs[0][:, ch]))
        axs_q[ch, 0].plot(freqs, 10*np.log10(real_qs[1][:, ch]))
        axs_q[ch, 0].plot(freqs, 10*np.log10(real_qs[2][:, ch]))
        axs_q[ch, 0].grid()

        axs_m[ch].plot(freqs, 10*np.log10(real_qs[1][:, ch]), label='Real')
        axs_m[ch].grid()

        axs_q[ch, 1].plot(freqs, 10*np.log10(fake_qs[0][:, ch]))
        axs_q[ch, 1].plot(freqs, 10*np.log10(fake_qs[1][:, ch]))
        axs_q[ch, 1].plot(freqs, 10*np.log10(fake_qs[2][:, ch]))
        axs_q[ch, 1].grid()

        axs_m[ch].plot(freqs, 10*np.log10(fake_qs[1][:, ch]), label='Synthetic')

        for i, stage in enumerate(stages):
            if ch in (2, 3):
                _freqs = freqs_lt_8hz
            else:
                _freqs = freqs

            line_real, = axs_s[i, ch].plot(_freqs, 10*np.log10(real_qs_stage[stage][1][:len(_freqs), ch]), label='Real',
                              linestyle='dashed', linewidth=2.5)
            line_fake, = axs_s[i, ch].plot(_freqs, 10*np.log10(fake_qs_stage[stage][1][:len(_freqs), ch]), label='Synthetic',
                              linewidth=2.5)
            if ch == 0:
                axs_s[i, ch].text(-0.25, 0.5, stages[i], fontsize=14, #fontweight='bold',
                                transform=axs_s[i, ch].transAxes, horizontalalignment='center')
            axs_s[i, ch].grid()

        axs_q[ch, 0].text(0.5, 0.8, input_names[ch], fontsize=12,
                          transform=axs_q[ch, 0].transAxes, horizontalalignment='center')
        axs_m[ch].text(0.5, 0.8, input_names[ch], fontsize=12,
                       transform=axs_m[ch].transAxes, horizontalalignment='center')
        axs_s[0, ch].text(0.5, 1.1, input_names[ch], fontsize=14, fontweight='bold',
                          transform=axs_s[0, ch].transAxes, horizontalalignment='center')

        if ch == nchs - 1:
            axs_m[ch].legend(loc='lower left', bbox_to_anchor=(-0.005, -0.03), fontsize=12)
        
    fig_s.legend(handles=[line_real, line_fake], loc='outside upper left', fontsize=15)
    fig_s.supxlabel('Frequency (Hz)', fontsize=15)#, fontweight='bold')
    fig_s.supylabel('Power (dB/Hz)', fontsize=15)#, fontweight='bold')

    axs_q[-1, 0].set_xlabel('Frequency (Hz)', fontsize=12)
    axs_q[-1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
    axs_m[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    axs_q[0, 0].set_ylabel('Power (dB/Hz)', fontsize=12)
    axs_m[0].set_ylabel('Power (dB/Hz)', fontsize=12)
    
    fig_q.savefig(fig_savedir / 'psd_quantiles_slf.pdf', dpi=300, bbox_inches='tight')
    fig_m.savefig(fig_savedir / 'psd_medians_slf.pdf', dpi=300, bbox_inches='tight')
    fig_s.savefig(fig_savedir / 'psd_medians_stages.pdf', dpi=300, bbox_inches='tight')

    return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=Path, required=True)
    parser.add_argument('--fig-savedir', type=Path, required=True)
    parser.add_argument('--copy-dst', type=Path, default=None)
    parser.add_argument('--cache-path', type=Path, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args.fig_savedir.mkdir(parents=True, exist_ok=True)
    plot_median_psds_from_slf_ds(
        cfg_path=args.cfg_path,
        fig_savedir=args.fig_savedir,
        copy_dst=args.copy_dst,
        cache_path=args.cache_path
    )
    logger.info('Done.')
