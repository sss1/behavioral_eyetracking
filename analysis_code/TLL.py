"""
This module implements and validates the trial log-likelihood (TLL) statistic
for assessing HMM fit, as discussed in Section 5.3 of the paper.
"""
import numpy as np
import numpy.ma as ma
import pickle
import math
import csv

from hmm import log_emission_prob
from sessions_list import get_sessions

# BEGIN CODE FOR SPECIFYING/LOADING DATA
trials_to_show = range(1, 11)
conditions_to_use = 'Both' # Optionally include only sessions of a given condition; should be one of 'Both', 'shrinky', or 'noshrinky'
sessions_to_use = get_sessions(conditions_to_use)
num_sessions = len(sessions_to_use)
sigma = 300
with open(str(sigma) + '.pickle', 'rb') as f:
  subjects = pickle.load(f)

# BEGIN CODE FOR LIKELIHOOD ANALYSIS
def compute_TLL(gaze, object_positions):
  return np.nanmean([log_emission_prob(E, X, sigma2 = sigma**2) for (E, X) in zip(gaze, object_positions)])

TLL_statistics = []
prop_off_tasks = []
for (subject_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):
  for trial_idx in trials_to_show:

    # Extract relevant HMM predictions, object positions, and gaze positions
    # Downsample by factor of 6 to compare with human coding
    eyetrack_trial = subjects[subject_ID].experiments[experiment].datatypes['eyetrack'].trials[trial_idx]
    trackit_trial = subjects[subject_ID].experiments[experiment].datatypes['trackit'].trials[trial_idx]
    trial_HMM = eyetrack_trial.HMM_MLE[::6]
    object_positions = trackit_trial.object_positions[:, ::6, :]
    predicted_object_positions = np.array([object_positions[trial_HMM[n], n, :] for n in range(len(trial_HMM))])
    gaze_positions = eyetrack_trial.data[::6, 1:3]

    # Extract relevant human codings
    human_coding_filename = '../human_coded/' + coder + '/' + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_coding.csv'
    with open(human_coding_filename, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader, None) # Skip CSV header line
      total_off_task_frames = 0
      total_frames = 0
      for (n, line) in enumerate(reader):
        total_frames += 1
        if line[1] == 'Off Task':
          total_off_task_frames += 1
      if(total_frames > 0):
        # Compute proportion of ``Off Task'' frames and TLL statistic for trial
        prop_off_tasks.append(float(total_off_task_frames)/total_frames)
        TLL_statistics.append(compute_TLL(gaze_positions, predicted_object_positions))

# Compute actual correlation between proportion of ``Off Task'' frames and TLL statistic
def get_masked_corr_coef(a, b):
  return ma.corrcoef(ma.masked_invalid(a), ma.masked_invalid(b))[0,1]
r = get_masked_corr_coef(prop_off_tasks, TLL_statistics)

# Compute 95% confidence interval based on asymptotic normality of Fisher Z-transformation of r
z = 0.5*math.log((1+r)/(1-r))
z_radius = 1.96/math.sqrt(len(TLL_statistics))
z_lower = z - z_radius
z_upper = z + z_radius
r_lower = (math.exp(2*z_lower) - 1)/((math.exp(2*z_lower) + 1))
r_upper = (math.exp(2*z_upper) - 1)/((math.exp(2*z_upper) + 1))

# Compute 95% confidence interval based on 10000 bootstrap resamples of trials
# Returns a bootstrapped resample (with the same sample size) from the pairs of lists a and b
def get_yoked_bootstrap_sample(a, b):
  return zip(*[(a[i], b[i]) for i in np.random.choice(len(a), len(a))])
bootstrapped_corrs = [get_masked_corr_coef(*get_yoked_bootstrap_sample(prop_off_tasks, TLL_statistics)) for _ in range(10000)]
bootstrapped_lower = np.quantile(bootstrapped_corrs, 0.025)
bootstrapped_higher = np.quantile(bootstrapped_corrs, 0.975)

print('Correlation: ' + str(r))
print('Fisher-transformed 95% CI: (' + str(r_lower) + ', ' + str(r_upper) + ')')
print('Bootstrapped 95% CI: ' + '(' + str(bootstrapped_lower) + ', ' + str(bootstrapped_higher) + ')')
