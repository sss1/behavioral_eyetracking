"""Prints participant demographics."""

from create_subjects_csvs import get_experiment_data

df = get_experiment_data()
print('Participant age: M = {}, SD = {}'.format(df['age'].mean(),
                                                df['age'].std()))
print('Proportion Male: {}'.format((df['sex'] == 'Male').sum()))
print('Proportion Female: {}'.format((df['sex'] == 'Female').sum()))
print('Proportion UNKNOWN sex: {}'.format((df['sex'] == 'UNKNOWN').sum()))
