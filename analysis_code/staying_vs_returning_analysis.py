import itertools
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
np.set_printoptions(suppress=True)

# Load all subject data
sigma = 300
fname = '../cache/' + str(sigma) + '.pickle'
print('Loading data from file ' + fname + '...')
with open(fname, 'rb') as f:
  subjects = pickle.load(f, encoding='latin1')

# Remove subjects with too much missing data
def subject_is_good(subject):
  return (len(subject.experiments['shrinky'].trials_to_keep) >= 5 and
        len(subject.experiments['noshrinky'].trials_to_keep) >= 5)
subjects = [subject for subject in subjects.values() if subject_is_good(subject)]
num_subjects = len(subjects)

print('Loaded {} good subjects.'.format(num_subjects))

def compute_trial_PTDT(object_seq):
  transitions_from_distractor_to_target = 0
  total_transitions_from_distractor = 0
  for first, second in zip(object_seq, object_seq[1:]):
    if first > 0 and second != first and second >= 0:
      total_transitions_from_distractor += 1
      if second == 0:
        transitions_from_distractor_to_target += 1
  if total_transitions_from_distractor > 0:
    return transitions_from_distractor_to_target/total_transitions_from_distractor
  else:
    return float('nan')

def compute_trial_NDT(object_seq):
  group_lengths = [(g[0], len(list(g[1]))) for g in itertools.groupby(trial_HMM)]
  mean_on_target_group_length = np.mean([l[1] for l in group_lengths if l[0] == 0])
  mean_nonmissing_group_length = np.mean([l[1] for l in group_lengths if l[0] >= 0])
  return (mean_on_target_group_length - mean_nonmissing_group_length)/60

experiment_mean_PTDTs = {'shrinky' : [], 'noshrinky' : []}
experiment_mean_NDTs = {'shrinky' : [], 'noshrinky' : []}
for subject in subjects:
  for experiment in ['shrinky', 'noshrinky']:

    # Use all trials except practice trial (trial 0)
    trials_to_show = range(int(subject
        .experiments[experiment]
        .datatypes['trackit']
        .metadata['Trial Count']))[1:]

    experiment_PTDTs = []
    experiment_NDTs = []
    for trial_idx in trials_to_show:
      trial_HMM = (subject
          .experiments[experiment]
          .datatypes['eyetrack']
          .trials[trial_idx]
          .HMM_MLE)

      experiment_PTDTs.append(compute_trial_PTDT(trial_HMM))
      experiment_NDTs.append(compute_trial_NDT(trial_HMM))

    experiment_mean_PTDTs[experiment].append(np.nanmean(experiment_PTDTs))
    experiment_mean_NDTs[experiment].append(np.nanmean(experiment_NDTs))

plt.figure()
plt.subplot(2, 2, 1)
plt.hist(experiment_mean_PTDTs['shrinky'])
plt.title('shrinky PTDT')
plt.subplot(2, 2, 3)
plt.hist(experiment_mean_PTDTs['noshrinky'])
plt.title('noshrinky PTDT')
plt.subplot(2, 2, 2)
plt.hist(experiment_mean_NDTs['shrinky'])
plt.title('shrinky NDT')
plt.subplot(2, 2, 4)
plt.hist(experiment_mean_NDTs['noshrinky'])
plt.title('noshrinky NDT')

plt.figure()
plt.subplot(2, 2, 1)
plt.scatter([subject.experiments['shrinky'].age for subject in subjects], experiment_mean_PTDTs['shrinky'])
plt.title('shrinky PTDT over age')
plt.subplot(2, 2, 3)
plt.scatter([subject.experiments['noshrinky'].age for subject in subjects], experiment_mean_PTDTs['noshrinky'])
plt.title('noshrinky PTDT over age')
plt.subplot(2, 2, 2)
plt.scatter([subject.experiments['shrinky'].age for subject in subjects], experiment_mean_NDTs['shrinky'])
plt.title('shrinky NDT over age')
plt.subplot(2, 2, 4)
plt.scatter([subject.experiments['noshrinky'].age for subject in subjects], experiment_mean_NDTs['noshrinky'])
plt.title('noshrinky NDT over age')
plt.show()
