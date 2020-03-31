import pandas as pd
import pickle

from stats_utils import experiment_loc_acc, experiment_ptdt, experiment_ndt, experiment_pfot, experiment_atd, experiment_atr, experiment_wtd, experiment_btd
metrics_list = [experiment_loc_acc, experiment_ptdt, experiment_ndt, experiment_pfot, experiment_atd, experiment_atr, experiment_wtd, experiment_btd]

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

def calc_metrics(subjects):
  table_as_dict = {'condition': [], 'age': [], 'subject_id': []}
  for metric in metrics_list:
    metric_name = metric.__name__[11:]
    table_as_dict[metric_name] = []

  for subject in subjects:
    for condition in _CONDITIONS:
      table_as_dict['subject_id'].append(subject.ID)
      table_as_dict['condition'].append(condition)
      experiment = subject.experiments[condition]
      table_as_dict['age'].append(experiment.age)
      for metric in metrics_list:
        metric_name = metric.__name__[11:]
        table_as_dict[metric_name].append(metric(experiment))
  return pd.DataFrame(table_as_dict)

def main():
  subjects = load_subjects()
  df = calc_metrics(subjects)


if __name__ == '__main__':
  main()
