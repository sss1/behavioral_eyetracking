"""Statistical helper functions for staying_vs_returning.py."""

from enum import Enum, auto
import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

from typing import Callable, Collection, List, NamedTuple, Tuple

class MissingDataTreatment(Enum):
  AVERAGE_CASE = auto()
  WORST_CASE = auto()

_TRIALS_TO_KEEP = list(range(1, 11))

Run = NamedTuple('Run', [('object', int), ('length', int)])

def average_over_trials(metric: Callable, experiment):
  """Computes the average of a metric over trial."""
  return np.nanmean(
          [metric(experiment.datatypes['eyetrack'].trials[trial_idx])
           for trial_idx in _TRIALS_TO_KEEP])

def experiment_loc_acc(experiment):
  return np.mean(
      [(experiment
        .datatypes['trackit']
        .trials[trial_idx]
        .trial_metadata['gridClickCorrect'] == 'true')
       for trial_idx in _TRIALS_TO_KEEP])

def experiment_ptdt(experiment, omit_missing_frames=True) -> float:
  def trial_ptdt(trial):
    """Computes proportion of transitions from distractors to target (PTDT)."""
    frames = trial.HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]

    transitions_from_distractor_to_target = 0
    transitions_from_distractor = 0
    for first, second in zip(frames, frames[1:]):
      if first > 0 and second != first and second >= 0:
        transitions_from_distractor += 1
        if second == 0:
          transitions_from_distractor_to_target += 1
    try:
      return transitions_from_distractor_to_target \
              /transitions_from_distractor
    except ZeroDivisionError:
      return float('nan')
  return average_over_trials(trial_ptdt, experiment)

def calc_run_lengths(sequence: List[int]) -> List[Run]:
  """Computes lengths of contiguous runs of the same object."""
  return [Run(object=g[0], length=len(list(g[1])))
          for g in itertools.groupby(sequence)]

def experiment_ndt(experiment, omit_missing_frames=True) -> float:
  def trial_ndt(trial):
    """Computes normalized duration on target (NDT) in seconds."""
    frames = trial.HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]

    run_lengths = calc_run_lengths(frames)
    mean_on_target_run_length = np.mean(
        [l.length for l in run_lengths if l.object == 0])
    mean_run_length = np.mean([l.length for l in run_lengths])

    return (mean_on_target_run_length - mean_run_length)/60
  return average_over_trials(trial_ndt, experiment)

def experiment_pfot(experiment, omit_missing_frames=True) -> float:
  def trial_pfot(trial):
    """Computes proportion of frames on target (PFT)."""
    frames = trial.HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]
    return np.mean(frames == 0)
  return average_over_trials(trial_pfot, experiment)

def experiment_atd(experiment, omit_missing_frames=True) -> float:
  def trial_atd(trial):
    """Computes average tracking duration (ATD) in seconds."""
    frames = trial.HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]
    total_frames = len(frames)
    num_runs = len([run for run in calc_run_lengths(frames)])
    if num_runs == 0:
      return float('nan')
    return (total_frames/num_runs)/60
  return average_over_trials(trial_atd, experiment)

def experiment_atr(experiment, omit_missing_frames=True) -> float:
  def trial_atr(trial):
    """Computes average time to return (ATR) in seconds."""
    frames = trial.HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]

    runs = calc_run_lengths(trial.HMM_MLE)
    return_times = []
    current_return_time = 0
    for run in runs:
      if run.object == 0:
        return_times.append(current_return_time/60)
        current_return_time = 0
      else:
        current_return_time += run.length
    return np.mean(return_times)

  return average_over_trials(trial_atr, experiment)

def experiment_wtd(experiment, omit_missing_frames=True) -> float:
  def trial_wtd(trial):
    """Computes within-trial decrement (WDT)."""
    if omit_missing_frames:
      x = np.arange(len(trial.HMM_MLE))[trial.HMM_MLE >= 0]
      y = (trial.HMM_MLE[trial.HMM_MLE >= 0] == 0)
    else:
      x = np.array(len(trial.HMM_MLE))
      y = (trial.HMM_MLE == 0)
    return linear_regression_with_CIs(x, y, return_CIs=False)
  return average_over_trials(trial_wtd, experiment)

def experiment_btd(experiment, omit_missing_frames=True) -> float:
  """Computes between-trials decrement (BTD).
  
  Note that, unlike the other performance metrics, BTD can only be computed at
  the experiment level, not at the trial level.
  """
  trial_pfots = []
  for trial_idx in _TRIALS_TO_KEEP:
    frames = experiment.datatypes['eyetrack'].trials[trial_idx].HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]
    trial_pfots.append(np.mean(frames == 0))
  zipped = [(x, y) for (x, y) in zip(_TRIALS_TO_KEEP, trial_pfots)
            if not math.isnan(y)]
  xs, ys = zip(*zipped)
  slope = linear_regression_with_CIs(xs, ys, return_CIs=False)
  return slope

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

def linear_regression_with_CIs(x, y, return_CIs = True):
  if len(x) < 2:
    if return_CIs:
      return float('nan'), float('nan'), float('nan')
    return float('nan')

  model = sm.OLS(y, sm.add_constant(x)).fit()
  if not return_CIs:
    return model.params[1]
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
