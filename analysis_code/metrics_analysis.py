import matplotlib.pyplot as plt
import pandas as pd
from create_subjects_csvs import get_experiment_data
import statsmodels.formula.api as smf
import seaborn as sns

def main():

  df = get_experiment_data()
  rename_map = {'loc_acc': 'Location Accuracy',
                'pfot': 'Proportion Of Frames On Target',
                'staying': 'Staying',
                'returning': 'Returning',
                'wtd': 'Within-Trials Decrement',
                'btd': 'Between-Trials Decrement',
                'atd': 'Average Tracking Duration',
                'atr': 'Average Time To Return',
                'proportion_missing_eyetracking': 'Proportion Missing'}
  measures = rename_map.keys()

  plt.figure()
  plt.suptitle('Performance Measures over Age')
  for idx, measure in enumerate(measures, 1):
    print(smf.ols(formula='{} ~ age'.format(measure), data=df).fit().summary())
    plt.subplot(3, 3, idx)
    ax = sns.regplot(x='age', y=measure, data=df[df['condition'] == 'shrinky'], units='subject_id')
    ax = sns.regplot(x='age', y=measure, data=df[df['condition'] == 'noshrinky'], units='subject_id')
    ax.set(xlabel='Age (years)', ylabel=rename_map[measure])

  plt.figure()
  plt.suptitle('Performance Measures over Location Accuracy')
  for idx, measure in enumerate(measures, 1):
    if idx == 1:
      continue
    print(smf.ols(formula='{} ~ loc_acc'.format(measure), data=df).fit().summary())
    plt.subplot(3, 3, idx)
    ax = sns.regplot(x='loc_acc', y=measure, data=df[df['condition'] == 'shrinky'], units='subject_id')
    ax = sns.regplot(x='loc_acc', y=measure, data=df[df['condition'] == 'noshrinky'], units='subject_id')
    ax.set(xlabel='Location Accuracy', ylabel=rename_map[measure])

  plt.show()


if __name__ == '__main__':
  main()
