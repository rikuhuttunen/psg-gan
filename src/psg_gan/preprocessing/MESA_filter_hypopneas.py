"""In the MESA dataset, the hypopneas have been scored whether or not they relate
to desaturations or arousals. Implement the MESA scoring rules for linking the events
and filter out irrelevant hypopneas.
"""
import argparse
import logging
import pandas as pd
import sleeplab_format as slf

from pathlib import Path
from sleeplab_format.models import AASMEvent, Annotation
from tqdm import tqdm


logger = logging.getLogger(__name__)


def filter_hypopneas(
        annotations: list[Annotation[AASMEvent]],
        spo2_arr: slf.models.SampleArray,
        min_desat: float = 3.0,
        desat_window: float = 45.0,
        arousal_window: float | None = 5.0) -> list[Annotation[AASMEvent]]:
    """Filter out hypopneas that have less than `min_desat` percentage points between
    the maximum saturation during the event and minimum saturation within
    `desat_window` seconds after the event. Keep the hypopneas when an arousal starts
    during the event or within `arousal_window` seconds after the event."""
    # Sort the annotations by start second
    annotations = sorted(annotations, key=lambda a: a.start_sec)

    # Create a dataframe from arousals to look up when linking hypopneas to arousals
    arousal_list = [dict(a) for a in annotations if a.name.startswith('AROUSAL')]
    if len(arousal_list) > 0:
        arousal_df = pd.DataFrame(arousal_list)
    else:
        arousal_df = None
    
    filtered_annotations = []
    for ann in annotations:
        # Keep all annotations other than hypopneas untouched
        if ann.name not in (AASMEvent.HYPOPNEA, AASMEvent.HYPOPNEA_CENTRAL, AASMEvent.HYPOPNEA_OBSTRUCTIVE):
            filtered_annotations.append(ann)
            continue

        # Find the first ongoing desaturation after the event within desat_window
        # UPDATE: The desaturations seem to be quite arbitrarily scored, so just take the
        # max saturation during the event and min saturation within desat_window after event
        fs = spo2_arr.attributes.sampling_rate
        during_win_start_idx = int(ann.start_sec * fs)
        during_win_end_idx = int((ann.start_sec + ann.duration) * fs)
        during_window = spo2_arr.values_func()[during_win_start_idx:during_win_end_idx]
        after_win_end_idx = int((ann.start_sec + ann.duration + desat_window) * fs)
        after_window = spo2_arr.values_func()[during_win_end_idx:after_win_end_idx]
        try:
            desat = round(float(max(during_window) - min(after_window)), 2)
        except ValueError as e:
            print(ann)
            print(e)
            continue

        # Check whether an arousal started during or after the event
        is_arousal_linked = False
        if arousal_window is not None and arousal_df is not None:
            aro_win_end_sec = ann.start_sec + ann.duration + arousal_window
            linked_arousals = arousal_df[(arousal_df.start_sec > ann.start_sec) & (arousal_df.start_sec < aro_win_end_sec)]
            is_arousal_linked = len(linked_arousals) > 0

        # Filter out the event if all conditions were not met
        if desat < min_desat and not is_arousal_linked:
            continue

        if ann.extra_attributes is None:
            ann.extra_attributes = {'spo2_desat': desat, 'is_arousal_linked': is_arousal_linked}
        else:
            ann.extra_attributes['spo2_desat'] = desat
            ann.extra_attributes['is_arousal_linked'] = is_arousal_linked

        filtered_annotations.append(ann)

    return filtered_annotations


def process_dataset(
        ds_dir: Path,
        series_names: list[str],
        annotation_name: str,
        new_scorer_name: str,
        spo2_key: str,
        min_desat: float = 3.0,
        desat_window: float = 45.0,
        arousal_window: float | None = 5.0,
        save_dir: Path | None = None) -> None:
    """Filter out hypopneas that have less than `min_desat` percentage points between
    the maximum saturation during the event and minimum saturation within
    `desat_window` seconds after the event. Keep the hypopneas when an arousal starts
    during the event or within `arousal_window` seconds after the event."""
    logger.info(f'Reading dataset from {ds_dir}')
    ds = slf.reader.read_dataset(ds_dir)
    for series_name in series_names:
        logger.info(f'Processing series {series_name}...')
        for subject_id, subject in tqdm(ds.series[series_name].subjects.items()):
            filtered_annotations = filter_hypopneas(
                annotations=subject.annotations[annotation_name].annotations,
                spo2_arr=subject.sample_arrays[spo2_key],
                min_desat=min_desat,
                desat_window=desat_window,
                arousal_window=arousal_window
            ) 

            # Write the annotations
            _write_subject = slf.models.Subject(
                metadata=subject.metadata,
                annotations={f'{new_scorer_name}_aasmevents': slf.models.AASMEvents(
                    scorer=new_scorer_name,
                    annotations=filtered_annotations
                )}
            )
            if save_dir is not None:
                save_subject_path = save_dir / subject_id
                save_subject_path.mkdir(exist_ok=True)
            else:
                save_subject_path = ds_dir / series_name / subject_id
            slf.writer.write_annotations(
                subject=_write_subject,
                subject_path=save_subject_path,
                format='json'
            )

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-dir', type=Path, required=True)
    parser.add_argument('--series-names', type=str, nargs='+')
    parser.add_argument('--annotation-name', type=str, required=True)
    parser.add_argument('--new-scorer-name', type=str, required=True,
        help='The name of the filtered annotations to be saved')
    parser.add_argument('--spo2-key', type=str, required=True)
    parser.add_argument('--min-desat', type=float, default=3.0,
        help='Minimum desaturation for a hypopnea to be included')
    parser.add_argument('--desat-window', type=float, default=45.0,
        help='The time window in seconds after event for desat linking')
    parser.add_argument('--arousal-window', type=float, default=5.0,
        help='The time window in seconds after event for arousal linking')
    parser.add_argument('--save-dir', type=Path, default=None)
    args = parser.parse_args()

    logger.info('Start processing dataset...')
    process_dataset(**vars(args))
    logger.info('Done.')
