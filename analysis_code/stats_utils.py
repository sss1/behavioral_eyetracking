"""Statistical helper functions for staying_vs_returning.py."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

def compute_ptdt(trial):
  """Computes proportion of transitions from distractors to target."""
  transitions_from_distractor_to_target = 0
  transitions_from_distractor = 0
  for first, second in zip(trial.HMM_MLE, trial.HMM_MLE[1:]):
    if first > 0 and second != first and second >= 0:
      transitions_from_distractor += 1
      if second == 0:
        transitions_from_distractor_to_target += 1
  try:
    return transitions_from_distractor_to_target \
            /transitions_from_distractor
  except ZeroDivisionError:
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

def report_ttest_1sample(null_hypothesis, sample, popmean, alpha=0.05):
  """Pretty-prints results of a two-sided one-sample t-test."""

  t_value, p_value = stats.ttest_1samp(sample, popmean)
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('Sample mean: {}, Sample SD: {}'.format(np.mean(sample), np.std(sample)))
  print('t({})={}, p={}.'.format(len(sample)-1, t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def report_ttest_paired_2sample(null_hypothesis, sample1, sample2, alpha=0.05):
  """Pretty-prints results of a two-sided paired two-sample t-test."""

  t_value, p_value = stats.ttest_rel(sample1, sample2)
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('Sample 1 mean: {}, Sample 1 SD: {}'.format(np.mean(sample1), np.std(sample1)))
  print('Sample 2 mean: {}, Sample 2 SD: {}'.format(np.mean(sample2), np.std(sample2)))
  print('t({})={}, p={}.'.format(len(sample1)-1, t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def linreg_summary_and_plot(x, y, condition, name=None, plot=True):
  if name:
    print('{} ({}):\n'.format(name.upper(), condition.upper()))
  xs = x[condition]
  ys = y[condition]
  print(sm.OLS(ys, sm.add_constant(xs)).fit().summary())
  if plot:
    sns.regplot(xs, ys, truncate=False)

def linear_regression_with_CIs(x, y):
  model = sm.OLS(y, sm.add_constant(x)).fit()
  CI = model.conf_int(alpha=0.05, cols=[1])[0]
  return model.params[1], CI[0], CI[1]

def _calc_indirect_effect(x, y, m, return_prop=False):
  x = stats.zscore(x)
  y = stats.zscore(y)
  m = stats.zscore(m)
  direct_effect = sm.OLS(y, sm.add_constant(x)).fit().params[1]
  xs = np.stack((x, m), axis=1)
  remaining_effect = sm.OLS(y, sm.add_constant(xs)).fit().params[1]
  indirect_effect = direct_effect - remaining_effect
  if return_prop:
    proportion_mediated = 1 - remaining_effect/direct_effect
    return indirect_effect, proportion_mediated
  return indirect_effect

def mediation_analysis(x, y, m, condition, title, num_reps = 10000):
  n = len(x[condition])
  indirect_effect, prop_mediated = _calc_indirect_effect(x[condition],
                                                         y[condition],
                                                         m[condition],
                                                         return_prop=True)
  subsampled_indirect_effects = np.zeros((num_reps,))
  for rep in range(num_reps):
    samples = random.choices(range(n), k=n)
    x_sub = [x[condition][i] for i in samples]
    y_sub = [y[condition][i] for i in samples]
    m_sub = [m[condition][i] for i in samples]

    subsampled_indirect_effects[rep] = _calc_indirect_effect(x_sub, y_sub,
                                                             m_sub)
  CI_lower = np.percentile(subsampled_indirect_effects, 2.5)
  CI_upper = np.percentile(subsampled_indirect_effects, 97.5)
  p_value = np.mean(subsampled_indirect_effects < 0)
  print(title)
  print('Indirect Effect: {}    95% CI: ({}, {})'.format(indirect_effect, CI_lower, CI_upper))
  print('Proportion Mediation: {}'.format(prop_mediated))
  print('p-value: {}'.format(p_value))
