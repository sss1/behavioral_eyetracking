import numpy as np
import pickle

import load_subjects as ls
from eyetracking_hmm import performance_according_to_HMM
from util import impute_missing_data, break_eyetracking_into_trials, interpolate_trackit_to_eyetracking, filter_experiment

sigma = 300
sigma2 = sigma ** 2

# Output pickle file to which to save results
save_file = str(sigma) + '.pickle'

# Load all experiment data
subjects = ls.load_dataset('shrinky', 'eyetrack')
subjects = ls.load_dataset('shrinky', 'trackit', subjects)
subjects = ls.load_dataset('noshrinky', 'eyetrack', subjects)
subjects = ls.load_dataset('noshrinky', 'trackit', subjects)

print('Merging and preprocessing datasets...')
# Combine eyetracking with trackit data and perform all preprocessing
for subject in subjects.values():
  for (experiment_ID, experiment) in subject.experiments.items():
    impute_missing_data(experiment)
    break_eyetracking_into_trials(experiment)
    interpolate_trackit_to_eyetracking(experiment)
    filter_experiment(experiment)

# Retain only subjects with at least half non-missing data in at least half their trials, in both conditions
def subject_is_good(subject):
  return len(subject.experiments['shrinky'].trials_to_keep) >= 5 and len(subject.experiments['noshrinky'].trials_to_keep) >= 5

# Filter out subjects with too much missing data
good_subjects = { subject_ID : subject for (subject_ID, subject) in subjects.items() if subject_is_good(subject) }
print(str(len(good_subjects)) + ' good subjects: ' + str(good_subjects.keys()))
bad_subjects = set(subjects.keys()) - set(good_subjects.keys())
print(str(len(bad_subjects)) + ' bad subjects: ' + str(bad_subjects))

for subject in good_subjects.values():
  for trial in zip(subject.experiments['shrinky'].datatypes['trackit'].trials,
                   subject.experiments['shrinky'].datatypes['eyetrack'].trials):
    performance_according_to_HMM(*trial, sigma2=sigma2)
  for trial in zip(subject.experiments['noshrinky'].datatypes['trackit'].trials,
                   subject.experiments['noshrinky'].datatypes['eyetrack'].trials):
    performance_according_to_HMM(*trial, sigma2=sigma2)

# Save results with Pickle
with open(save_file , 'wb') as f:
  pickle.dump(subjects, f)
print('Saved subjects to file \''  + save_file + '\'')
