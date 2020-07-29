import pandas as pd
import pickle

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
  subjects = [subject for subject in subjects.values()
              if subject_is_good(subject)]
  print('Loaded {} good subjects.\n'.format(len(subjects)))
  return subjects

def subject_is_good(subject):
  """Whether a subject satisfies inclusion criteria."""
  return (len(subject.experiments['shrinky'].trials_to_keep) >= 5 and
          len(subject.experiments['noshrinky'].trials_to_keep) >= 5)


def get_frame_data():
  subjects = load_subjects()
  table_as_dict = {
      'subject_id': [], 'age': [], 'condition': [], 'trial_num': [],
      'trial_len': [], 'loc_acc': [], 'error_type': [], 'frame': [], 'HMM': []
  }

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in _TRIALS_TO_SHOW:
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]
        for frame, HMM in enumerate(eyetrack_trial.HMM_MLE):

          # Experiment-level data
          table_as_dict['subject_id'].append(subject.ID)
          table_as_dict['condition'].append(experiment.ID)
          table_as_dict['age'].append(experiment.age)

          # Trial-level data
          table_as_dict['trial_num'].append(trial_idx)
          table_as_dict['trial_len'].append(len(eyetrack_trial.HMM_MLE))
          table_as_dict['loc_acc'].append(
              trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
          table_as_dict['error_type'].append(
              trackit_trial.trial_metadata['errorType'])

          # Frame-level data
          table_as_dict['frame'].append(frame)
          table_as_dict['HMM'].append(HMM)

  return pd.DataFrame(table_as_dict)
  

def get_trial_data():
  subjects = load_subjects()
  table_as_dict = {'subject_id': [], 'age': [], 'condition': [], 'target': [],
                   'trial_num': [], 'trial_len': [], 'loc_acc': [], 'pfot': [],
                   'returning': [], 'staying': [],
                   'proportion_missing_eyetracking': [], 'atr': [], 'wtd': [],
                   'atd': []}

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in _TRIALS_TO_SHOW:
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]

        # Experiment-level data
        table_as_dict['subject_id'].append(subject.ID)
        table_as_dict['condition'].append(experiment.ID)
        table_as_dict['age'].append(experiment.age)

        # Trial-level data
        table_as_dict['trial_num'].append(trial_idx)
        table_as_dict['trial_len'].append(len(eyetrack_trial.HMM_MLE))
        table_as_dict['target'].append(trackit_trial.trial_metadata['target'])
        table_as_dict['loc_acc'].append(
            trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
        table_as_dict['pfot'].append(stats_utils.trial_pfot(eyetrack_trial))
        table_as_dict['returning'].append(stats_utils.trial_ptdt(eyetrack_trial))
        table_as_dict['staying'].append(stats_utils.trial_ndt(eyetrack_trial))
        table_as_dict['atd'].append(stats_utils.trial_atd(eyetrack_trial))
        table_as_dict['wtd'].append(stats_utils.trial_wtd(eyetrack_trial))
        table_as_dict['atr'].append(stats_utils.trial_atr(eyetrack_trial))
        table_as_dict['proportion_missing_eyetracking'].append(
            eyetrack_trial.proportion_missing)

  return pd.DataFrame(table_as_dict)
  

def get_experiment_data():
  subjects = load_subjects()
  table_as_dict = {'subject_id': [], 'age': [], 'btd': [], 'condition': [],
                   'loc_acc': [], 'pfot': [], 'returning': [],
                   'staying': [], 'proportion_missing_eyetracking': [],
                   'atr': [], 'wtd': [], 'atd': []}

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in _TRIALS_TO_SHOW:
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]

        # Experiment-level data
        table_as_dict['subject_id'].append(subject.ID)
        table_as_dict['condition'].append(experiment.ID)
        table_as_dict['age'].append(experiment.age)
        table_as_dict['btd'].append(stats_utils.experiment_btd(experiment))

        # Trial-level data
        table_as_dict['loc_acc'].append(
            trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
        table_as_dict['pfot'].append(stats_utils.trial_pfot(eyetrack_trial))
        table_as_dict['returning'].append(stats_utils.trial_ptdt(eyetrack_trial))
        table_as_dict['staying'].append(stats_utils.trial_ndt(eyetrack_trial))
        table_as_dict['atd'].append(stats_utils.trial_atd(eyetrack_trial))
        table_as_dict['wtd'].append(stats_utils.trial_wtd(eyetrack_trial))
        table_as_dict['atr'].append(stats_utils.trial_atr(eyetrack_trial))
        table_as_dict['proportion_missing_eyetracking'].append(
            eyetrack_trial.proportion_missing)

  return (pd.DataFrame(table_as_dict)
            .groupby(by=['subject_id', 'condition'])
            .mean()
            .reset_index())
  

def main():
  # print(get_frame_data())
  # print(get_trial_data())
  print(get_experiment_data())

  

if __name__ == '__main__':
  main()
