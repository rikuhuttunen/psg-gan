import argparse
import jax
import logging
import numpy as np
import sleeplab_format as slf

from ast import literal_eval
from datetime import timedelta
from flax import nnx
from orbax.checkpoint import v1 as ocp
from pathlib import Path
from psg_gan.utime import models, training
from sleeplab_format.extractor import preprocess
from tqdm import tqdm


logger = logging.getLogger(__name__)


def parse_hypnogram(arr, start_ts, scorer, epoch_sec=30.0) -> slf.models.Hypnogram:
    stage_map = {
        0: slf.models.AASMSleepStage.W,
        1: slf.models.AASMSleepStage.N1,
        2: slf.models.AASMSleepStage.N2,
        3: slf.models.AASMSleepStage.N3,
        4: slf.models.AASMSleepStage.R
    }
    annotations = []
    pred = np.array(np.argmax(arr, axis=1)).astype(int)
    for i, stage_int in enumerate(pred):
        start_delta = i * timedelta(seconds=epoch_sec)
        stage = slf.models.Annotation[slf.models.AASMSleepStage](
            name=stage_map[stage_int],
            start_ts=start_ts + start_delta,
            start_sec=float(start_delta.seconds),
            duration=epoch_sec
        )
        annotations.append(stage)

    return slf.models.Hypnogram(annotations=annotations, scorer=scorer)


def classify_events(arr: np.array, no_event_threshold: float = 0.5):
    # First define which is higher: apnea or hypopnea confidence
    res = np.argmax(arr[..., 1:], axis=1) + 1  # +1 so that we have 1s and 2s
    # Then, set the indices where the confidence of no event is higher than the threshold
    res[arr[..., 0] > no_event_threshold] = 0
    
    return res


def get_continuous_segments(arr, val_arr=[1,2], patience=0, min_event_length=11):
    """Compute start and stop idx of each segment in samples belonging to val_arr.
    
    Args:
        patience: the number of changed value in samples before ending an event.
            e.g. if fs=1 and patience=2, there can be two seconds scored differently,
            while still considered as the same event.
        cond_f: a function which takes one argument, the array segment of interest;
            if evaluated to False the event will be discarded
    """
    is_apnea = lambda arr: sum(arr == 2) >= min_event_length
    segs = []
    event_on = False
    
    if patience is None:
        patience = 0
        
    if min_event_length is None:
        min_event_length = 1
    
    for i, v in enumerate(arr):
        if v in val_arr:
            if not event_on:
                start = i
                event_on = True
        else:
            if event_on:
                if not np.any(np.isin(arr[max(0, i-patience):i], val_arr)):
                    stop = i-1-patience
                    event_on = False
                    
                    # Discard too short events
                    if stop - start + 1 < min_event_length:
                        continue

                    if is_apnea(arr[start:stop+1]):
                        etype = slf.models.AASMEvent.APNEA
                    else:
                        etype = slf.models.AASMEvent.HYPOPNEA

                    segs.append((start, stop, etype))

    return segs


def parse_events(arr, start_ts, scorer, no_event_threshold=0.5) -> slf.models.AASMEvents:
    events = get_continuous_segments(classify_events(arr, no_event_threshold=no_event_threshold))
    annotations = []
    
    for start, stop, etype in events:
        event = slf.models.Annotation[slf.models.AASMEvent](
            name=etype,
            start_ts=start_ts + timedelta(seconds=start),
            start_sec=float(start),
            duration = float(stop - start + 1)
        )
        annotations.append(event)

    return slf.models.AASMEvents(annotations=annotations, scorer=scorer)


def parse_annotations(scorer_outputs, start_ts, scorer='utime', no_event_threshold=0.5) -> slf.models.BaseAnnotations:
    res = {}
    if 'hypnogram' in scorer_outputs.keys():
        res[f'{scorer}_hypnogram'] = parse_hypnogram(np.array(scorer_outputs['hypnogram'][0, ...]), start_ts, scorer)
    
    if 'events' in scorer_outputs.keys():
        res[f'{scorer}_aasmevents'] = parse_events(np.array(scorer_outputs['events'][0, ...]), start_ts, scorer, no_event_threshold=no_event_threshold)

    return res


def score_slf_subject(
        subject: slf.models.Subject,
        model: models.UTime,
        fs: float,
        scorer_name: str,
        input_name_map: dict[str, str] | None = None
        ) -> dict[str, slf.models.BaseAnnotations]:
    """Score the subject with model.
    
    Returns:
        The hypnogram and events in sleeplab-format
    """
    inputs = {}
    for input_name, sarr_name in input_name_map.items():
        sarr = subject.sample_arrays[sarr_name]
        s = sarr.values_func()
        if sarr.attributes.sampling_rate != fs:
            s = preprocess.resample_polyphase(s, sarr.attributes, fs_new=fs)
        inputs[input_name] = s[np.newaxis, ..., np.newaxis]

    logits = model(inputs)
    outputs = {k: jax.nn.softmax(v) for k, v in logits.items()}

    start_ts = subject.metadata.recording_start_ts
    annotations = parse_annotations(outputs, start_ts, scorer=scorer_name)

    return annotations


def score_slf_dataset(
        ds: slf.models.Dataset,
        model: nnx.Module,
        scorer_name: str,
        fs: float,
        input_name_map: str):
    annotation_series = {}
    for series_name, series in ds.series.items():
        annotation_subjects = {}
        for sid, subj in tqdm(series.subjects.items()):
            # logger.info(f'Scoring subject {sid}')
            annotations = score_slf_subject(subj, model, fs, scorer_name, input_name_map)
            annotation_subjects[sid] = slf.models.Subject(
                metadata=subj.metadata,
                annotations=annotations
            )
        annotation_series[series_name] = slf.models.Series(name=series_name, subjects=annotation_subjects)

    return slf.models.Dataset(name=ds.name, series=annotation_series)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds-dir', type=Path, required=True)
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--scorer-name', type=str, required=True)
    parser.add_argument('--fs', type=float, required=True)
    parser.add_argument('--ds-save-dir', type=Path, default=None)
    parser.add_argument('--input-name-map', type=str,
        default="{'c4m1': 'C4M1_64Hz_hp', 'e1fpz': 'E1M2_64Hz_hp', 'naspres': 'NasalPressure_64Hz', 'ppg':'Pleth_64Hz'}",
    help='Mapping from checkpoint input names to SLF dataset signal names')

    args = parser.parse_args()

    logger.info(f'Reading dataset from {args.ds_dir}')
    ds = slf.reader.read_dataset(args.ds_dir)

    logger.info(f'Restoring checkpoint from {args.ckpt_dir}')
    with ocp.training.Checkpointer(args.ckpt_dir) as ckptr:
        ckpt = training.restore_checkpoint(ckptr, restore_ema=True)

    annotation_ds = score_slf_dataset(
        ds=ds,
        model=ckpt['model'],
        scorer_name=args.scorer_name,
        fs=args.fs,
        input_name_map=literal_eval(args.input_name_map)
    )

    if args.ds_save_dir is not None:
        ds_save_dir = Path(args.ds_save_dir)
        annotation_ds.name = ds_save_dir.name
    else:
        ds_save_dir = Path(args.ds_dir)

    slf.writer.write_dataset(annotation_ds, basedir=Path(ds_save_dir).parent)
    logger.info('Autoscoring finished.')    
