"""This module implements analyses comparing staying and returning."""
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
sns.set()
np.set_printoptions(suppress=True)

import create_subjects_csvs
import stats_utils

# Load all subject data
_SIGMA = 300
_TRIALS_TO_SHOW = range(1, 11) # use all trials except practice trial (trial 0)
_CONDITIONS = ['shrinky', 'noshrinky']
_FNAME = '../cache/' + str(_SIGMA) + '.pickle'


def load_subjects():
  """Load and filter subjects from pickle file."""
  print('Loading data from file ' + _FNAME + '...')
  with open(_FNAME, 'rb') as input_file:
    subjects = pickle.load(input_file, encoding='latin1')
  subjects = [subject
              for subject in subjects.values() if subject_is_good(subject)]
  print('Loaded {} good subjects.\n'.format(len(subjects)))
  return subjects

def subject_is_good(subject):
  """Whether a subject satisfies inclusion criteria."""
  return (len(subject.experiments['shrinky'].trials_to_keep) >= 5 and
          len(subject.experiments['noshrinky'].trials_to_keep) >= 5)

def extract_experiment_stats(subjects, experiment_func):
  return {condition : [experiment_func(subject.experiments[condition])
                       for subject in subjects]
          for condition in _CONDITIONS}

def compute_statistics(subjects):
  """Computes mean PTDT and NDT for each experiment."""
  for subject in subjects:
    for condition in ['shrinky', 'noshrinky']:
      experiment = subject.experiments[condition]

      experiment.mean_loc_acc = stats_utils.experiment_loc_acc(experiment)
      experiment.mean_ptdt = stats_utils.experiment_ptdt(experiment)
      experiment.mean_ndt = stats_utils.experiment_ndt(experiment)

  ages = extract_experiment_stats(subjects, lambda exp : exp.age)
  loc_accs = extract_experiment_stats(subjects, lambda exp : exp.mean_loc_acc)
  ptdts = extract_experiment_stats(subjects, lambda exp : exp.mean_ptdt)
  ndts = extract_experiment_stats(subjects, lambda exp : exp.mean_ndt)

  return ages, loc_accs, ptdts, ndts

def report_statistics_and_make_plots(ages, loc_accs, ptdts, ndts):
  """Generates statistics and plots reported in paper."""

  sessions_df = create_subjects_csvs.get_experiment_data()
  shrinky_df = sessions_df[sessions_df['condition'] == 'shrinky']
  noshrinky_df = sessions_df[sessions_df['condition'] == 'noshrinky']

  # Compare statistics to chance values
  stats_utils.report_ttest_1sample(null_hypothesis="mean(shrinky PTDT) == 1/6",
                                   sample=shrinky_df['returning'], popmean=1/6)
  stats_utils.report_ttest_1sample(
      null_hypothesis="mean(noshrinky PTDT) == 1/6",
      sample=noshrinky_df['returning'], popmean=1/6)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(shrinky NDT) == 0",
                       sample=shrinky_df['staying'], popmean=0)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(noshrinky NDT) == 0",
                       sample=noshrinky_df['staying'], popmean=0)

  # Compare statistics between conditions
  stats_utils.report_ttest_2sample(
      null_hypothesis="mean(shrinky location accuracy) == mean(noshrinky location accuracy)",
      sample1=shrinky_df['loc_acc'], sample2=noshrinky_df['loc_acc'],
      paired=True)
  stats_utils.report_ttest_2sample(
      null_hypothesis="mean(shrinky PTDT) == mean(noshrinky PTDT)",
      sample1=shrinky_df['returning'], sample2=noshrinky_df['returning'],
      paired=True)
  stats_utils.report_ttest_2sample(
      null_hypothesis="mean(shrinky NDT) == mean(noshrinky NDT)",
      sample1=shrinky_df['staying'], sample2=noshrinky_df['staying'],
      paired=True)

  stats_utils.linreg_summary_and_plot(x='age', y='loc_acc', data=shrinky_df, name='Location Accuracy over Age', plot=False)
  stats_utils.linreg_summary_and_plot(x='age', y='loc_acc', data=noshrinky_df, name='Location Accuracy over Age', plot=False)

  # Linearly regress statistics over age
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((0, 1))
  plt.plot([3.5, 6], [1/6, 1/6], c='red', ls='--')
  plt.ylabel('PTDT')
  plt.title('Returning, Salient Target')
  stats_utils.linreg_summary_and_plot(x='age', y='returning', data=shrinky_df, name='PTDT over age')

  plt.subplot(2, 2, 2)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((0, 1))
  plt.plot([3.5, 6], [1/6, 1/6], c='red', ls='--')
  plt.title('Returning, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(x='age', y='returning', data=noshrinky_df, name='PTDT over age')

  plt.subplot(2, 2, 3)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.plot([3.5, 6], [0, 0], c='red', ls='--')
  plt.ylabel('NDT (seconds)')
  plt.title('Staying, Salient Target')
  stats_utils.linreg_summary_and_plot(x='age', y='staying', data=shrinky_df, name='NDT over age')

  plt.subplot(2, 2, 4)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.title('Staying, Non-Salient Target')
  plt.plot([3.5, 6], [0, 0], c='red', ls='--')
  stats_utils.linreg_summary_and_plot(x='age', y='staying', data=noshrinky_df, name='NDT over age')
  plt.tight_layout()

  plt.savefig('figs/linear_regressions_over_age.pdf')

  # Linearly regress statistics over location accuracy
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((0, 1))
  plt.plot([0, 1], [1/6, 1/6], c='red', ls='--')
  plt.ylabel('Returning')
  plt.title('Returning, Salient Target')
  stats_utils.linreg_summary_and_plot(x='loc_acc', y='returning', data=shrinky_df,
                                      name='PTDT over location accuracy')
  plt.subplot(2, 2, 2)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((0, 1))
  plt.plot([0, 1], [1/6, 1/6], c='red', ls='--')
  plt.ylabel('Returning')
  plt.title('Returning, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(x='loc_acc', y='returning', data=noshrinky_df,
                                      name='PTDT over location accuracy')
  plt.subplot(2, 2, 3)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((-0.4, 2.4))
  plt.plot([0, 1], [0, 0], c='red', ls='--')
  plt.ylabel('Staying')
  plt.title('Staying, Salient Target')
  stats_utils.linreg_summary_and_plot(x='loc_acc', y='staying', data=shrinky_df,
                                      name='NDT over location accuracy')
  plt.subplot(2, 2, 4)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((-0.4, 2.4))
  plt.plot([0, 1], [0, 0], c='red', ls='--')
  plt.ylabel('Staying (seconds)')
  plt.title('Staying, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(x='loc_acc', y='staying', data=noshrinky_df,
                                      name='NDT over location accuracy')
  plt.tight_layout()
  plt.savefig('figs/linear_regressions_over_location_accuracy.pdf')

  # Mediation Analysis
  stats_utils.mediation_analysis(x='age', y='loc_acc', m='returning', data=shrinky_df,
                                 title='PTDT mediating effect of Age on Loc Acc, shrinky')
  stats_utils.mediation_analysis(x='age', y='loc_acc', m='returning', data=noshrinky_df,
                                 title='PTDT mediating effect of Age on Loc Acc, noshrinky')
  stats_utils.mediation_analysis(x='age', y='loc_acc', m='staying', data=shrinky_df,
                                 title='NDT mediating effect of Age on Loc Acc, shrinky')
  stats_utils.mediation_analysis(x='age', y='loc_acc', m='staying', data=noshrinky_df,
                                 title='NDT mediating effect of Age on Loc Acc, noshrinky')

  y = np.concatenate((stats.rankdata(shrinky_df['returning']),
                      stats.rankdata(noshrinky_df['returning']),
                      stats.rankdata(shrinky_df['staying']),
                      stats.rankdata(noshrinky_df['staying'])))
  x_age = np.concatenate((shrinky_df['age'], noshrinky_df['age'],
                          shrinky_df['age'], noshrinky_df['age']))
  x_measure_type = np.concatenate((np.zeros_like(shrinky_df['returning']),
                                   np.zeros_like(noshrinky_df['returning']),
                                   np.ones_like(shrinky_df['staying']),
                                   np.ones_like(noshrinky_df['staying'])))
  x_interaction = np.multiply(x_age, x_measure_type)
  X = np.stack((x_age, x_measure_type, x_interaction), axis=1)
  print(sm.OLS(y, sm.add_constant(X)).fit().summary())

  plt.show()

def main():
  """Performs and reports analyses comparing staying and returning."""
  report_statistics_and_make_plots(*compute_statistics(load_subjects()))

if __name__ == '__main__':
  main()
