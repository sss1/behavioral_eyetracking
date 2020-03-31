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
  # Compare statistics to chance values
  stats_utils.report_ttest_1sample(null_hypothesis="mean(shrinky PTDT) == 1/6",
                                   sample=ptdts['shrinky'], popmean=1/6)
  stats_utils.report_ttest_1sample(
      null_hypothesis="mean(noshrinky PTDT) == 1/6",
      sample=ptdts['noshrinky'], popmean=1/6)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(shrinky NDT) == 0",
                       sample=ndts['shrinky'], popmean=0)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(noshrinky NDT) == 0",
                       sample=ndts['noshrinky'], popmean=0)

  # Compare statistics between conditions
  stats_utils.report_ttest_paired_2sample(
      null_hypothesis="mean(shrinky location accuracy) == mean(noshrinky location accuracy)",
      sample1=loc_accs['shrinky'], sample2=loc_accs['noshrinky'])
  stats_utils.report_ttest_paired_2sample(
      null_hypothesis="mean(shrinky PTDT) == mean(noshrinky PTDT)",
      sample1=ptdts['shrinky'], sample2=ptdts['noshrinky'])
  stats_utils.report_ttest_paired_2sample(
      null_hypothesis="mean(shrinky NDT) == mean(noshrinky NDT)",
      sample1=ndts['shrinky'], sample2=ndts['noshrinky'])

  stats_utils.linreg_summary_and_plot(ages, loc_accs, 'shrinky', 'Location Accuracy over Age', plot=False)
  stats_utils.linreg_summary_and_plot(ages, loc_accs, 'noshrinky', 'Location Accuracy over Age', plot=False)

  # Linearly regress statistics over age
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((0, 1))
  plt.ylabel('PTDT')
  plt.title('Returning, Salient Target')
  stats_utils.linreg_summary_and_plot(ages, ptdts, 'shrinky', 'PTDT over age')

  plt.subplot(2, 2, 2)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((0, 1))
  plt.title('Returning, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(ages, ptdts, 'noshrinky', 'PTDT over age')

  plt.subplot(2, 2, 3)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.ylabel('NDT (seconds)')
  plt.title('Staying, Salient Target')
  stats_utils.linreg_summary_and_plot(ages, ndts, 'shrinky', 'NDT over age')

  plt.subplot(2, 2, 4)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.title('Staying, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(ages, ndts, 'noshrinky', 'NDT over age')
  plt.tight_layout()

  plt.savefig('figs/linear_regressions_over_age.pdf')

  # Linearly regress statistics over location accuracy
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((0, 1))
  plt.ylabel('PTDT')
  plt.title('Returning, Salient Target')
  stats_utils.linreg_summary_and_plot(loc_accs, ptdts, 'shrinky',
                                      'PTDT over location accuracy')
  plt.subplot(2, 2, 2)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((0, 1))
  plt.title('Returning, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(loc_accs, ptdts, 'noshrinky',
                                      'PTDT over location accuracy')
  plt.subplot(2, 2, 3)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((-0.4, 2.4))
  plt.ylabel('NDT (seconds)')
  plt.title('Staying, Salient Target')
  stats_utils.linreg_summary_and_plot(loc_accs, ndts, 'shrinky',
                                      'NDT over location accuracy')
  plt.subplot(2, 2, 4)
  plt.xlim((0, 1))
  plt.xlabel('Location Accuracy')
  plt.ylim((-0.4, 2.4))
  plt.title('Staying, Non-Salient Target')
  stats_utils.linreg_summary_and_plot(loc_accs, ndts, 'noshrinky',
                                      'NDT over location accuracy')
  plt.tight_layout()
  plt.savefig('figs/linear_regressions_over_location_accuracy.pdf')

  # Mediation Analysis
  stats_utils.mediation_analysis(ages, loc_accs, ptdts, 'shrinky',
                                 'PTDT mediating effect of Age on Loc Acc, shrinky')
  stats_utils.mediation_analysis(ages, loc_accs, ptdts, 'noshrinky',
                                 'PTDT mediating effect of Age on Loc Acc, noshrinky')
  stats_utils.mediation_analysis(ages, loc_accs, ndts, 'shrinky',
                                 'NDT mediating effect of Age on Loc Acc, shrinky')
  stats_utils.mediation_analysis(ages, loc_accs, ndts, 'noshrinky',
                                 'NDT mediating effect of Age on Loc Acc, noshrinky')

  y = np.concatenate((stats.rankdata(ptdts['shrinky']),
                      stats.rankdata(ptdts['noshrinky']),
                      stats.rankdata(ndts['shrinky']),
                      stats.rankdata(ndts['noshrinky'])))
  x_age = np.concatenate((ages['shrinky'], ages['noshrinky'],
                          ages['shrinky'], ages['noshrinky']))
  x_measure_type = np.concatenate((np.zeros_like(ptdts['shrinky']),
                                   np.zeros_like(ptdts['noshrinky']),
                                   np.ones_like(ndts['shrinky']),
                                   np.ones_like(ndts['noshrinky'])))
  x_interaction = np.multiply(x_age, x_measure_type)
  X = np.stack((x_age, x_measure_type, x_interaction), axis=1)
  print(sm.OLS(y, sm.add_constant(X)).fit().summary())

  plt.show()

def main():
  """Performs and reports analyses comparing staying and returning."""
  report_statistics_and_make_plots(*compute_statistics(load_subjects()))

if __name__ == '__main__':
  main()
