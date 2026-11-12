"""Compare performance between manual and automatic scoring on a SLF dataset."""
import argparse
import json
import logging
import numpy as np
import pandas as pd
import sleeplab_format as slf

from pathlib import Path
from psg_gan.metrics.classification import plot_metrics
from sleeplab_format.models import AASMEvent, AASMSleepStage

logger = logging.getLogger(__name__)


STAGE_MAP = {
    AASMSleepStage.W: 0,
    AASMSleepStage.N1: 1,
    AASMSleepStage.N2: 2,
    AASMSleepStage.N3: 3,
    AASMSleepStage.R: 4
}

EVENT_MAP = {
    AASMEvent.HYPOPNEA: 1,
    AASMEvent.HYPOPNEA_CENTRAL: 1,
    AASMEvent.HYPOPNEA_OBSTRUCTIVE: 1,
    AASMEvent.APNEA: 2,
    AASMEvent.APNEA_CENTRAL: 2,
    AASMEvent.APNEA_MIXED: 2,
    AASMEvent.APNEA_OBSTRUCTIVE: 2
}


def annotations_to_vec(
        ann_list: list[slf.models.Annotation],
        start_sec: float,
        end_sec: float,
        fs: float,
        val_map: dict[str, int],
        default: int = 0) -> np.array:
    """Transform lists of SLF Annotations to equally sampled lists of ints."""
    # Use pandas dataframes for processing
    df = pd.DataFrame([dict(a) for a in ann_list])
    full_duration = end_sec - start_sec

    if len(df) > 0:
        df['end_sec'] = df['start_sec'] + df['duration']

        # Filter out events outside the desired timeframe
        df = df[df['end_sec'] > start_sec]
        df = df[df['start_sec'] < end_sec]

        # Filter out keys not in val_map
        df = df[df['name'].isin(val_map.keys())]

        # Transform start and end times from seconds to indices in evenly sampled result
        df['start_idx'] = ((df['start_sec'] - start_sec) * fs).astype(int)
        df['start_idx'] = df['start_idx'].clip(lower=0)

        df['end_idx'] = ((df['end_sec'] - start_sec) * fs).astype(int)
        df['end_idx'] = df['end_idx'].clip(upper=int(fs*full_duration))

    # Create evenly sampled numpy array
    res = np.full(int(full_duration * fs), default)
    for _, row in df.iterrows():
        res[row.start_idx:row.end_idx] = val_map[row['name']]

    return res


def plot_confmats(
        ds: slf.models.Dataset,
        save_dir: Path,
        manual_scorer: str,
        auto_scorer: str,
        epoch_sec: float = 30.0,
        hg_cm_figsize=(7, 6),
        event_cm_figsize=(5, 4.5),
        save_suffix='') -> None:
    """Plot confusion matrices between manual and automatic scoring separately for all series."""
    all_manual_stages = []
    all_auto_stages = []
    all_manual_events = []
    all_auto_events = []

    for series in ds.series.values():
        logger.info(f'Computing metrics for series {series.name}...')
        series_manual_stages = []
        series_auto_stages = []
        series_manual_events = []
        series_auto_events = []

        for subject in series.subjects.values():
            man_hg_list = subject.annotations[f'{manual_scorer}_hypnogram'].annotations
            auto_hg_list = subject.annotations[f'{auto_scorer}_hypnogram'].annotations
            
            # Resolve start and end for the hypnogram from the manual scoring
            start_sec = man_hg_list[0].start_sec
            end_sec = man_hg_list[-1].start_sec + man_hg_list[-1].duration
            # Temporary hack until STAGES SLF converter is fixed...
            i = 1
            while start_sec > end_sec:
                start_sec = man_hg_list[i].start_sec
                i += 1
            
            man_event_list = subject.annotations[f'{manual_scorer}_aasmevents'].annotations
            auto_event_list = subject.annotations[f'{auto_scorer}_aasmevents'].annotations

            try:
                man_hg = annotations_to_vec(man_hg_list, start_sec, end_sec, fs=1/epoch_sec, val_map=STAGE_MAP)
                auto_hg = annotations_to_vec(auto_hg_list, start_sec, end_sec, fs=1/epoch_sec, val_map=STAGE_MAP)
                man_events = annotations_to_vec(man_event_list, start_sec, end_sec, fs=1.0, val_map=EVENT_MAP)
                auto_events = annotations_to_vec(auto_event_list, start_sec, end_sec, fs=1.0, val_map=EVENT_MAP)

            except ValueError as e:
                logger.warn(f'Skipping {subject.metadata.subject_id} due to following error:')
                logger.warn(e)
                continue

            series_manual_stages.append(man_hg)
            series_auto_stages.append(auto_hg)
            series_manual_events.append(man_events)
            series_auto_events.append(auto_events)

        series_manual_stages = np.concatenate(series_manual_stages)
        series_auto_stages = np.concatenate(series_auto_stages)
        all_manual_stages.append(series_manual_stages)
        all_auto_stages.append(series_auto_stages)
        
        series_manual_events = np.concatenate(series_manual_events)
        series_auto_events = np.concatenate(series_auto_events)
        all_manual_events.append(series_manual_events)
        all_auto_events.append(series_auto_events)

        plot_metrics(
            series_manual_stages,
            series_auto_stages,
            class_names=[k.name for k in STAGE_MAP.keys()],
            save_dir=save_dir,
            save_prefix=f'{series.name}_hg_{save_suffix}',
            cm_xlabel='Automatic',
            cm_ylabel='Manual',
            cm_figsize=hg_cm_figsize
        )
        plot_metrics(
            series_manual_events,
            series_auto_events,
            class_names=['No event', 'Hypopnea', 'Apnea'],
            save_dir=save_dir,
            save_prefix=f'{series.name}_events_{save_suffix}',
            cm_xlabel='Automatic',
            cm_ylabel='Manual',
            cm_figsize=event_cm_figsize
        )
    
    # Plot metrics for all data combined
    if len(ds.series) > 1:
        all_manual_stages = np.concatenate(all_manual_stages)
        all_auto_stages = np.concatenate(all_auto_stages)
        plot_metrics(
            all_manual_stages,
            all_auto_stages,
            class_names=[k.name for k in STAGE_MAP.keys()],
            save_dir=save_dir,
            save_prefix=f'all_hg',
            cm_xlabel='Automatic',
            cm_ylabel='Manual',
            cm_figsize=hg_cm_figsize
        )

        all_manual_events = np.concatenate(all_manual_events)
        all_auto_events = np.concatenate(all_auto_events)
        plot_metrics(
            all_manual_events,
            all_auto_events,
            class_names=['No event', 'Hypopnea', 'Apnea'],
            save_dir=save_dir,
            save_prefix=f'all_events',
            cm_xlabel='Automatic',
            cm_ylabel='Manual',
            cm_figsize=event_cm_figsize
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-dir', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--manual-scorer', default='STAGES')
    parser.add_argument('--auto-scorer', default='utime')
    parser.add_argument('--subject-id-path', default=None)
    parser.add_argument('--save-suffix', default='')
    args = parser.parse_args()
    
    logger.info(f'Reading dataset from {args.ds_dir}')
    ds = slf.reader.read_dataset(Path(args.ds_dir))

    if args.subject_id_path is not None:
        logger.info(f'Including subject IDs from {args.subject_id_path}')
        with open(args.subject_id_path, 'r') as f:
            sids = json.load(f)
            for _ser_name, _sids in sids.items():
                _ser = ds.series[_ser_name]
                _new_subjects = {k: v for k, v in _ser.subjects.items() if k in _sids}
                ds.series[_ser_name].subjects = _new_subjects

    logger.info(f'Plotting confusion matrices...')
    plot_confmats(
        ds=ds,
        save_dir=Path(args.save_dir),
        manual_scorer=args.manual_scorer,
        auto_scorer=args.auto_scorer,
        save_suffix=args.save_suffix
    )
    logger.info('Done.')
