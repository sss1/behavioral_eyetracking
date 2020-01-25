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

def compute_ptdt(trial):
  """Computes proportion of transitions from distractors to target."""
  transitions_from_distractor_to_target = 0
  transitions_from_distractor = 0
  for first, second in zip(trial.HMM_MLE, trial.HMM_MLE[1:]):
    if first > 0 and second != first and second >= 0:
      transitions_from_distractor += 1
      if second == 0:
        transitions_from_distractor_to_target += 1
  if transitions_from_distractor > 0:
    return transitions_from_distractor_to_target \
            /transitions_from_distractor
  return float('nan')

def compute_ndt(trial):
  """Computes normalized duration on target."""
  group_lengths = [(g[0], len(list(g[1])))
                   for g in itertools.groupby(trial.HMM_MLE)]
  mean_on_target_group_length = np.mean(
      [l[1] for l in group_lengths if l[0] == 0])
  mean_nonmissing_group_length = np.mean(
      [l[1] for l in group_lengths if l[0] >= 0])
  return (mean_on_target_group_length - mean_nonmissing_group_length)/60

def compute_statistics(subjects):
  """Computes mean PTDT and NDT for each experiment."""
  for subject in subjects:
    for condition in ['shrinky', 'noshrinky']:
      experiment = subject.experiments[condition]

      experiment.mean_ptdt = np.nanmean(
          [compute_ptdt(experiment.datatypes['eyetrack'].trials[trial_idx])
           for trial_idx in _TRIALS_TO_SHOW])
      experiment.mean_ndt = np.nanmean(
          [compute_ndt(experiment.datatypes['eyetrack'].trials[trial_idx])
           for trial_idx in _TRIALS_TO_SHOW])

def report_ttest_1sample(null_hypothesis, sample, popmean, alpha=0.05):
  """Pretty-prints results of a two-sided one-sample t-test."""

  t_value, p_value = stats.ttest_1samp(sample, popmean)
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('T={}, p={}.'.format(t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def report_ttest_paired_2sample(null_hypothesis, sample1, sample2, alpha=0.05):
  """Pretty-prints results of a two-sided paired two-sample t-test."""

  t_value, p_value = stats.ttest_rel(sample1, sample2)
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('T={}, p={}.'.format(t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def report_statistics_and_make_plots(subjects):
  """Generates statistics and plots reported in paper."""
  ages = {condition : [subject.experiments[condition].age
                       for subject in subjects]
          for condition in _CONDITIONS}
  ptdts = {condition : [subject.experiments[condition].mean_ptdt
                        for subject in subjects]
           for condition in _CONDITIONS}
  ndts = {condition : [subject.experiments[condition].mean_ndt
                       for subject in subjects]
          for condition in _CONDITIONS}

  # Compare statistics to chance values
  report_ttest_1sample(null_hypothesis="mean(shrinky PTDT) == 1/6",
                       sample=ptdts['shrinky'], popmean=1/6)
  report_ttest_1sample(null_hypothesis="mean(noshrinky PTDT) == 1/6",
                       sample=ptdts['noshrinky'], popmean=1/6)
  report_ttest_1sample(null_hypothesis="mean(shrinky NDT) == 0",
                       sample=ndts['shrinky'], popmean=0)
  report_ttest_1sample(null_hypothesis="mean(noshrinky NDT) == 0",
                       sample=ndts['noshrinky'], popmean=0)

  # Compare statistics between conditions
  report_ttest_paired_2sample(
      null_hypothesis="mean(shrinky PTDT) == mean(noshrinky PTDT)",
      sample1=ptdts['shrinky'], sample2=ptdts['noshrinky'])
  report_ttest_paired_2sample(
      null_hypothesis="mean(shrinky NDT) == mean(noshrinky NDT)",
      sample1=ndts['shrinky'], sample2=ndts['noshrinky'])

  print(ptdts['shrinky'])
  print(ndts['shrinky'])

  # Linearly regress statistics over age
  print(min(ages['shrinky']), max(ages['shrinky']))
  print(min(ages['noshrinky']), max(ages['noshrinky']))
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.xlim((3.5, 6))
  plt.ylim((0, 1))
  plt.ylabel('PTDT')
  plt.gca().axes.get_xaxis().set_ticklabels([])
  plt.title('Shrinky PTDT')
  x = ages['shrinky'] # pylint: disable=invalid-name
  y = ptdts['shrinky'] # pylint: disable=invalid-name
  print(sm.OLS(y, sm.add_constant(x)).fit().summary())
  sns.regplot(x, y, truncate=False)

  plt.subplot(2, 2, 2)
  plt.xlim((3.5, 6))
  plt.ylim((0, 1))
  plt.gca().axes.get_xaxis().set_ticklabels([])
  plt.gca().axes.get_yaxis().set_ticklabels([])
  plt.title('Noshrinky PTDT')
  x = ages['noshrinky'] # pylint: disable=invalid-name
  y = ptdts['noshrinky'] # pylint: disable=invalid-name
  print(sm.OLS(y, sm.add_constant(x)).fit().summary())
  sns.regplot(x, y, truncate=False)

  plt.subplot(2, 2, 3)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.ylabel('NDT (seconds)')
  plt.title('Shrinky NDT')
  x = ages['shrinky'] # pylint: disable=invalid-name
  y = ndts['shrinky'] # pylint: disable=invalid-name
  print(sm.OLS(y, sm.add_constant(x)).fit().summary())
  sns.regplot(x, y, truncate=False)

  plt.subplot(2, 2, 4)
  plt.xlim((3.5, 6))
  plt.xlabel('Age (years)')
  plt.ylim((-0.4, 2.4))
  plt.gca().axes.get_yaxis().set_ticklabels([])
  plt.title('Noshrinky NDT')
  x = ages['noshrinky'] # pylint: disable=invalid-name
  y = ndts['noshrinky'] # pylint: disable=invalid-name
  print(sm.OLS(y, sm.add_constant(x)).fit().summary())
  sns.regplot(x, y, truncate=False)

  plt.savefig('figs/linear_regressions_over_age.pdf')

  plt.show()

def main():
  """Performs and reports analyses comparing staying and returning."""
  subjects = load_subjects()
  compute_statistics(subjects)
  report_statistics_and_make_plots(subjects)

if __name__ == '__main__':
  main()
