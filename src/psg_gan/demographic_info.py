"""Aggregate demographic information on STAGES, MESA, STAR-SA (ELIVE), and STAR-HV datasets."""
import argparse
import logging
import pandas as pd
import sleeplab_format as slf

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


STAGES_COL_MAP = {
    'subject_code': 'sid',
    'nsrr_age': 'age',
    'nsrr_sex': 'sex',
    'nsrr_bmi': 'bmi',
}

MESA_COL_MAP = {
    'mesaid': 'sid',
    'nsrr_age': 'age',
    'nsrr_sex': 'sex',
    'nsrr_bmi': 'bmi',
}

ELIVE_HV_COL_MAP = {
    'Tutkittavan ID': 'sid',
    'Ikä': 'age',
    'Sukupuoli': 'sex',
    'BMI': 'bmi'
}

IQR_PRETTY_COL_MAP = {
    'age': 'Age (years)',
    'bmi': 'Body mass index ($kg/m^2$)',
    'tst': 'Total sleep time (min)',
    'ahi': 'Apnea-hypopnea index (events/h)',
}

COUNT_COLS = [
    'Male',
    'Female',
    'No OSA',
    'Mild OSA',
    'Moderate OSA',
    'Severe OSA'
]


def calculate_metrics(subject: slf.models.Subject, hg_key: str, event_key: str) -> dict[str, Any]:
    hg_df = pd.DataFrame([dict(a) for a in subject.annotations[hg_key].annotations])
    event_df = pd.DataFrame([dict(a) for a in subject.annotations[event_key].annotations])

    # TST in minutes
    tst_min = sum(hg_df[hg_df['name'].isin(['N1', 'N2', 'N3', 'R'])]['duration']) / 60

    # AHI
    if len(event_df) > 0:
        n_apneas = len(event_df[event_df['name'].isin(['APNEA_OBSRTUCTIVE', 'APNEA_CENTRAL', 'APNEA_MIXED', 'APNEA'])])
        n_hypopneas = len(event_df[event_df['name'].isin(['HYPOPNEA_CENTRAL', 'HYPOPNEA_OBSTRUCTIVE', 'HYPOPNEA'])])
    else:
        n_apneas = 0
        n_hypopneas = 0

    ahi = (n_apneas + n_hypopneas) / (tst_min / 60)

    return {
        'tst': tst_min,
        'ahi': ahi,
        'No OSA': ahi < 5,
        'Mild OSA': 5 <= ahi < 15,
        'Moderate OSA': 15 <= ahi < 30,
        'Severe OSA': ahi > 30
    }


def compute_metrics(stages_df, stages_ds, mesa_df, mesa_ds, elive_df, elive_ds, hv_df, hv_ds,
                    stages_scorer, mesa_hypnogram_scorer, mesa_events_scorer, elive_scorer, hv_scorer):
    logger.info('Calculating metrics for STAGES...')
    stages_metric_dict = {}
    for idx in stages_df.index:
        stages_metric_dict[idx] = calculate_metrics(
            stages_ds.series['all'].subjects[idx],
            hg_key=f'{stages_scorer}_hypnogram',
            event_key=f'{stages_scorer}_aasmevents'
        )

    stages_metric_df = pd.DataFrame.from_dict(stages_metric_dict, orient='index').set_index(stages_df.index)
    stages_df = pd.concat([stages_df, stages_metric_df], axis=1)

    logger.info('Calculating metrics for MESA...')
    mesa_metric_dict = {}
    for idx in mesa_df.index:
        mesa_metric_dict[idx] = calculate_metrics(
            mesa_ds.series['psg'].subjects[idx],
            hg_key=f'{mesa_hypnogram_scorer}_hypnogram',
            event_key=f'{mesa_events_scorer}_aasmevents'
        )

    mesa_metric_df = pd.DataFrame.from_dict(mesa_metric_dict, orient='index').set_index(mesa_df.index)
    mesa_df = pd.concat([mesa_df, mesa_metric_df], axis=1)

    logger.info('Calculating metrics for ELIVE...')
    elive_metric_dict = {}
    for idx in elive_df.index:
        elive_metric_dict[idx] = calculate_metrics(
            elive_ds.series['PSG'].subjects[idx],
            hg_key=f'{elive_scorer}_hypnogram',
            event_key=f'{elive_scorer}_aasmevents'
        )

    elive_metric_df = pd.DataFrame.from_dict(elive_metric_dict, orient='index').set_index(elive_df.index)
    elive_df = pd.concat([elive_df, elive_metric_df], axis=1)

    logger.info('Calculating metrics for HV...')
    hv_metric_dict = {}
    for idx in hv_df.index:
        hv_metric_dict[idx] = calculate_metrics(
            hv_ds.series['PSG'].subjects[idx],
            hg_key=f'{hv_scorer}_hypnogram',
            event_key=f'{hv_scorer}_aasmevents'
        )

    hv_metric_df = pd.DataFrame.from_dict(hv_metric_dict, orient='index').set_index(hv_df.index)
    hv_df = pd.concat([hv_df, hv_metric_df], axis=1)

    return stages_df, mesa_df, elive_df, hv_df


def summarize_counts(df):
    count_df = df.sum().astype(int).to_frame()
    count_df.columns = ['Count (%)']
    count_df['Count (%)'] = count_df['Count (%)'].apply(lambda cnt: f'{cnt} ({(cnt / len(df) * 100):.1f}%)')

    return count_df


def summarize_iqr(df: pd.DataFrame) -> pd.Series:
    iqr_df = df[IQR_PRETTY_COL_MAP.keys()].describe().transpose()
    iqr_df['Median (IQR)'] = iqr_df[['25%', '50%', '75%']].apply(
        lambda quarts: f'{quarts.iloc[1]:.1f} ({quarts.iloc[0]:.1f}-{quarts.iloc[2]:.1f})', axis=1)
    iqr_df.index = IQR_PRETTY_COL_MAP.values()

    return iqr_df[['Median (IQR)']]


def summarize_demographics(stages_df, mesa_df, elive_df, hv_df):
    stages_iqr_df = summarize_iqr(stages_df)
    mesa_iqr_df = summarize_iqr(mesa_df)
    elive_iqr_df = summarize_iqr(elive_df)
    hv_iqr_df = summarize_iqr(hv_df)
    
    iqr_df = pd.concat([stages_iqr_df, mesa_iqr_df, elive_iqr_df, hv_iqr_df], axis=1)
    iqr_df.columns = [
        ['Median (IQR)', 'Median (IQR)', 'Median (IQR)', 'Median (IQR)'],
        ['STAGES', 'MESA', 'ELIVE', 'HV']]

    stages_df['Male'] = stages_df['sex'] == 'male'
    stages_df['Female'] = stages_df['sex'] == 'female'
    mesa_df['Male'] = mesa_df['sex'] == 'male'
    mesa_df['Female'] = mesa_df['sex'] == 'female'
    elive_df['Male'] = elive_df['sex'].isin(['Male', 'Mies'])
    elive_df['Female'] = elive_df['sex'].isin(['Female', 'Nainen'])
    hv_df['Male'] = hv_df['sex'].isin(['Male', 'Mies'])
    hv_df['Female'] = hv_df['sex'].isin(['Female', 'Nainen'])
    
    stages_count_df = summarize_counts(stages_df[COUNT_COLS])
    mesa_count_df = summarize_counts(mesa_df[COUNT_COLS])
    elive_count_df = summarize_counts(elive_df[COUNT_COLS])
    hv_count_df = summarize_counts(hv_df[COUNT_COLS])

    count_df = pd.concat([stages_count_df, mesa_count_df, elive_count_df, hv_count_df], axis=1)
    count_df.columns = [
        ['Count (%)', 'Count (%)', 'Count (%)', 'Count (%)'],
        ['STAGES', 'MESA', 'ELIVE', 'HV']]

    return iqr_df, count_df


def read_data(stages_ds_dir, mesa_ds_dir, elive_ds_dir, hv_ds_dir, stages_excel_path, mesa_excel_path, elive_hv_excel_path):
    logger.info(f'Reading STAGES data from {stages_ds_dir}')
    stages_ds = slf.reader.read_dataset(stages_ds_dir, include_annotations=True)
    # Merge all STAGES series into one to simplify further processing
    all_subjects = {}
    for sname in stages_ds.series:
        all_subjects.update(stages_ds.series[sname].subjects)
    stages_ds.series['all'] = slf.models.Series(name='all', subjects=all_subjects)

    logger.info(f'Reading MESA data from {mesa_ds_dir}')
    mesa_ds = slf.reader.read_dataset(mesa_ds_dir, include_annotations=True)

    logger.info(f'Reading ELIVE data from {elive_ds_dir}')
    elive_ds = slf.reader.read_dataset(elive_ds_dir, include_annotations=True)

    logger.info(f'Reading HV data from {hv_ds_dir}')
    hv_ds = slf.reader.read_dataset(hv_ds_dir, include_annotations=True)

    stages_sids = [subject.metadata.subject_id for subject in stages_ds.series['all'].subjects.values()]
    mesa_sids = [subject.metadata.subject_id for subject in mesa_ds.series['psg'].subjects.values()]
    elive_sids = [subject.metadata.subject_id for subject in elive_ds.series['PSG'].subjects.values()]
    hv_sids = [subject.metadata.subject_id for subject in hv_ds.series['PSG'].subjects.values()]

    logger.info(f'Reading the excel files to DataFrames...')
    stages_df = (pd.read_csv(stages_excel_path)
                 .rename(columns=STAGES_COL_MAP)
    )[list(STAGES_COL_MAP.values())]
    stages_df = stages_df[~stages_df['sid'].isna()].drop_duplicates(subset='sid').set_index('sid')

    mesa_df = (pd.read_csv(mesa_excel_path)
             .rename(columns=MESA_COL_MAP)
    )[list(MESA_COL_MAP.values())]
    mesa_df = mesa_df[~mesa_df['sid'].isna()]
    mesa_df['sid'] = mesa_df['sid'].apply(lambda i: f'mesa-sleep-{i:04d}')
    mesa_df = mesa_df.set_index('sid')

    elive_hv_df = (pd.read_excel(elive_hv_excel_path)
             .rename(columns=ELIVE_HV_COL_MAP)
    )[list(ELIVE_HV_COL_MAP.values())]
    elive_hv_df = elive_hv_df[~elive_hv_df['sid'].isna()]
    elive_hv_df['sid'] = elive_hv_df['sid'].astype(str)
    elive_hv_df = elive_hv_df.set_index('sid')

    return (stages_df.loc[stages_df.index.intersection(stages_sids)],
            stages_ds,
            mesa_df.loc[mesa_df.index.intersection(mesa_sids)],
            mesa_ds,
            elive_hv_df.loc[elive_hv_df.index.intersection(elive_sids)],
            elive_ds,
            elive_hv_df.loc[elive_hv_df.index.intersection(hv_sids)],
            hv_ds,
    )


def read_and_preprocess(
        stages_ds_dir: Path,
        mesa_ds_dir: Path,
        elive_ds_dir: Path,
        hv_ds_dir: Path,
        stages_excel_path: Path,
        mesa_excel_path: Path,
        elive_hv_excel_path: Path,
        stages_scorer: str,
        mesa_hypnogram_scorer: str,
        mesa_events_scorer: str,
        elive_scorer: str,
        hv_scorer: str,
        cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_dir is None:
        logger.info('Reading original data from disk...')
        stages_df, stages_ds, mesa_df, mesa_ds, elive_df, elive_ds, hv_df, hv_ds = read_data(
            stages_ds_dir, mesa_ds_dir, elive_ds_dir, hv_ds_dir, stages_excel_path, mesa_excel_path, elive_hv_excel_path)
        stages_df, mesa_df, elive_df, hv_df = compute_metrics(
            stages_df, stages_ds, mesa_df, mesa_ds, elive_df, elive_ds, hv_df, hv_ds,
            stages_scorer, mesa_hypnogram_scorer, mesa_events_scorer, elive_scorer, hv_scorer)
    else:
        try:
            logger.info(f'Reading cached dataframes from {cache_dir}')
            stages_df = pd.read_pickle(cache_dir / 'stages_df.pickle')
            mesa_df = pd.read_pickle(cache_dir / 'mesa_df.pickle')
            elive_df = pd.read_pickle(cache_dir / 'elive_df.pickle')
            hv_df = pd.read_pickle(cache_dir / 'hv_df.pickle')
        except FileNotFoundError:
            logger.info('Cached files not found, reading original data...')
            stages_df, stages_ds, mesa_df, mesa_ds, elive_df, elive_ds, hv_df, hv_ds = read_data(
                stages_ds_dir, mesa_ds_dir, elive_ds_dir, hv_ds_dir, stages_excel_path, mesa_excel_path, elive_hv_excel_path)
            stages_df, mesa_df, elive_df, hv_df = compute_metrics(
                stages_df, stages_ds, mesa_df, mesa_ds, elive_df, elive_ds, hv_df, hv_ds,
                stages_scorer, mesa_hypnogram_scorer, mesa_events_scorer, elive_scorer, hv_scorer)
            logger.info(f'Saving the dataframes to {cache_dir}')
            cache_dir.mkdir(exist_ok=True)
            stages_df.to_pickle(cache_dir / 'stages_df.pickle')
            mesa_df.to_pickle(cache_dir / 'mesa_df.pickle')
            elive_df.to_pickle(cache_dir / 'elive_df.pickle')
            hv_df.to_pickle(cache_dir / 'hv_df.pickle')
    
    return stages_df, mesa_df, elive_df, hv_df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages-ds-dir', type=Path, required=True)
    parser.add_argument('--mesa-ds-dir', type=Path, required=True)
    parser.add_argument('--star-sa-ds-dir', type=Path, required=True)
    parser.add_argument('--star-hv-ds-dir', type=Path, required=True)
    parser.add_argument('--stages-excel-path', type=Path, required=True)
    parser.add_argument('--mesa-excel-path', type=Path, required=True)
    parser.add_argument('--star-sa-hv-excel-path', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--stages-scorer', default='vaunted-trout-702')
    parser.add_argument('--mesa-hypnogram-scorer', default='nsrr')
    parser.add_argument('--mesa-events-scorer', default='hyp_desat3_nsrr')
    parser.add_argument('--star-sa-scorer', default='manually_corrected')
    parser.add_argument('--star-hv-scorer', default='best_weights_2021-06-14_162837.h5')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    stages_df, mesa_df, elive_df, hv_df = read_and_preprocess(
        stages_ds_dir=args.stages_ds_dir,
        mesa_ds_dir=args.mesa_ds_dir,
        elive_ds_dir=args.star_sa_ds_dir,
        hv_ds_dir=args.star_hv_ds_dir,
        stages_excel_path=args.stages_excel_path,
        mesa_excel_path=args.mesa_excel_path,
        elive_hv_excel_path=args.star_sa_hv_excel_path,
        stages_scorer=args.stages_scorer,
        mesa_hypnogram_scorer = args.mesa_hypnogram_scorer,
        mesa_events_scorer = args.mesa_events_scorer,
        elive_scorer=args.star_sa_scorer,
        hv_scorer=args.star_hv_scorer,
        cache_dir=Path(args.cache_dir) if args.cache_dir is not None else None,
    )
    iqr_df, count_df = summarize_demographics(stages_df, mesa_df, elive_df, hv_df)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    iqr_df.to_latex(buf=save_dir / 'demog_quantiles.tex')
    count_df.to_latex(buf=save_dir / 'demog_counts.tex')
