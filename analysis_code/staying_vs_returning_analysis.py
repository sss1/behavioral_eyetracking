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

print('Loaded {} good subjects.'.format(len(subjects)))

for subject in subjects:
  for experiment in ['shrinky', 'noshrinky']:

    # Use all trials except practice trial (trial 0)
    trials_to_show = range(int(subject
        .experiments[experiment]
        .datatypes['trackit']
        .metadata['Trial Count']))[1:]

  for trial_idx in trials_to_show:
    trial_HMM = (subject
        .experiments[experiment]
        .datatypes['eyetrack']
        .trials[trial_idx]
        .HMM_MLE)

