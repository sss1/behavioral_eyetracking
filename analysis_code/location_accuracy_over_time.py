import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import stats_utils

from typing import List

from create_subjects_csvs import get_frame_data

max_num_frames = 1200
max_num_secs = max_num_frames/60

def _get_corrs_over_time(df: pd.DataFrame, frame_column: str) -> List[float]:
  corrs = (df[['loc_acc',frame_column,'on_target']]
           .groupby(by=frame_column)
           .corr()
           .reset_index())
  return list(corrs[corrs['level_1'] == 'on_target']['loc_acc'])

def _get_on_target_encoding(HMM: int) -> int:
  if HMM == 0:
    return 1  # On-target
  elif HMM > 0:
    return -1  # Off-target
  return np.nan  # Missing data

def main():
  df = get_frame_data()

  df['on_target'] = df['HMM'].apply(_get_on_target_encoding)
  df['non_missing'] = df['HMM'] > -1
  df['frame_from_end'] = df['trial_len'] - df['frame']

  corrs_over_time = _get_corrs_over_time(df, 'frame')
  corrs_over_time_from_end = _get_corrs_over_time(df, 'frame_from_end')

  trials_over_time = (df
                      .groupby(by='frame')
                      .count()
                      .reset_index()['non_missing'])
  non_missing_over_time = (df[['frame', 'non_missing']]
                           .groupby(by='frame')
                           .sum()
                           .reset_index()['non_missing'])

  ts = [t/60 for t in range(max_num_frames)]

  # Correlation as a function of trial time
  ax = plt.subplot(3, 1, 1)
  plt.plot(ts, corrs_over_time[:max_num_frames])
  plt.xlim((0, max_num_secs))
  plt.ylim((0, 0.75))
  plt.ylabel('Correlation')
  plt.xlabel('Trial Time (seconds)')
  ax.text(1/6, 0.72, 'A', fontsize=12, fontweight='bold', va='top')
  print('Correlation at Trial-Onset: {}'.format(corrs_over_time[0]))
  print('Correlation at frame {}: {}'
        .format(max_num_frames, corrs_over_time[max_num_frames]))

  # Number of trials with nonmissing data as a function of trial time
  ax = plt.subplot(3, 1, 2)
  h1 = plt.plot(ts, trials_over_time[:max_num_frames])
  h2 = plt.plot(ts, non_missing_over_time[:max_num_frames], '--')
  plt.xlim((0, max_num_secs))
  plt.ylabel('Number of Trials')
  plt.xlabel('Trial Time (seconds)')
  ax.text(1/6, 175, 'B', fontsize=12, fontweight='bold', va='top')
  ax.legend(['Total Trials', 'Trials with Nonmissing Data'], loc=(0.1, 0.1))

  # Correlation as a function of time before trial end
  ax = plt.subplot(3, 1, 3)
  plt.plot(ts, corrs_over_time_from_end[:max_num_frames])
  plt.xlim((0, max_num_secs))
  plt.ylim((0, 0.75))
  plt.ylabel('Correlation')
  plt.xlabel('Time Before Trial End (seconds)')
  plt.tight_layout()
  ax.text(1/6, 0.13, 'C', fontsize=12, fontweight='bold', va='top')
  print('Correlation at Trial-Offset: {}'.format(corrs_over_time_from_end[0]))
  plt.show()

  print('Total proportion of missing data: {}'.format(1 - df['non_missing'].mean()))

  # How does eye-tracking at end of trial (EOT) relate to behavioral response?
  eot_df = df[df['frame_from_end'] == 1]
  print(eot_df
        .fillna(0)  # Code missing data as 0 to include in groupby
        .groupby(by=['on_target', 'error_type'])
        .count()
        ['subject_id']  # Arbitrarily pick the first column
        .reset_index()
        .pivot_table(index='error_type', values='subject_id', columns='on_target')
        .to_latex()
  )
  error_df = df[~df['loc_acc']]
  error_types = pd.get_dummies(error_df['error_type'])
  print(pd.concat([error_df[['on_target', 'loc_acc']], error_types], axis=1)
        .fillna(0)
        .groupby(by=['on_target', 'loc_acc'])
        .mean())

  # Test for covert attention; is mean accuracy on OffTarget EOT trials > chance?
  off_target_eot_location_accuracy = (eot_df[eot_df['on_target'] == -1]
                                      .groupby(by='subject_id')
                                      ['loc_acc']
                                      .mean()
                                      .values)
  chance = 1/36  # Probability of randomly selecting the target square
  stats_utils.report_ttest_1sample(
      null_hypothesis=f'mean(off_target_loc_acc) <= chance[{chance}]',
      sample=off_target_eot_location_accuracy,
      popmean=chance,
      one_sided=True)


if __name__ == '__main__':
  main()
