"""This script preprocesses the data and fits the HMM to each trial."""

import pickle

import load_subjects as ls
from eyetracking_hmm import performance_according_to_HMM
import util

_SIGMA = 300

# Output pickle file to which to save results
_SAVE_FILE = str(_SIGMA) + '.pickle'

# Load all experiment data
subjects = ls.load_dataset('shrinky', 'eyetrack')
subjects = ls.load_dataset('shrinky', 'trackit', subjects)
subjects = ls.load_dataset('noshrinky', 'eyetrack', subjects)
subjects = ls.load_dataset('noshrinky', 'trackit', subjects)

print('Merging and preprocessing datasets...')
# Combine eyetracking with trackit data and perform all preprocessing
for subject in subjects.values():
  for (experiment_ID, experiment) in subject.experiments.items():
    util.impute_missing_data(experiment)
    util.break_eyetracking_into_trials(experiment)
    util.interpolate_trackit_to_eyetracking(experiment)
    util.filter_experiment(experiment)

# Retain only subjects with at least half non-missing data in at least half
# their trials, in both conditions
def subject_is_good(subject):
  return (len(subject.experiments['shrinky'].trials_to_keep) >= 5
          and len(subject.experiments['noshrinky'].trials_to_keep) >= 5)

# Filter out subjects with too much missing data
good_subjects = {subject_ID : subject
                 for (subject_ID, subject) in subjects.items()
                 if subject_is_good(subject)}
print(str(len(good_subjects)) + ' good subjects: ' + str(good_subjects.keys()))
bad_subjects = set(subjects.keys()) - set(good_subjects.keys())
print(str(len(bad_subjects)) + ' bad subjects: ' + str(bad_subjects))

for subject in good_subjects.values():
  experiment = subject.experiments['shrinky']
  for trial in zip(experiment.datatypes['trackit'].trials,
                   experiment.datatypes['eyetrack'].trials):
    performance_according_to_HMM(*trial, sigma2=_SIGMA**2)
  experiment = subject.experiments['noshrinky']
  for trial in zip(experiment.datatypes['trackit'].trials,
                   experiment.datatypes['eyetrack'].trials):
    performance_according_to_HMM(*trial, sigma2=_SIGMA**2)

# Save results with Pickle
with open(_SAVE_FILE, 'wb') as fout:
  pickle.dump(subjects, fout)
print('Saved subjects to file \''  + _SAVE_FILE + '\'')
