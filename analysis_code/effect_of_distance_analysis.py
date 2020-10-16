# TODO: This script could be integrated into staying_vs_returning_analysis.py.
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skmisc.loess import loess
import seaborn as sns

import create_subjects_csvs
import stats_utils

# Number of TrackIt objects
_NUM_OBJECTS = 7

# Load frame-level data
df = create_subjects_csvs.get_frame_data()
df = df[df['condition'] == 'noshrinky']

# Compute next object for each pair of consecutive frames.
df = stats_utils.add_next_object_column(df)

# For computational reasons, we subsample non-transitions. Later we scale the
# regression response to account for this.
subsample_proportion = 0.001

distances_dict = {'distance': [], 'is_transition': [], 'is_from_target': [], 'is_to_target': []}
for idx, row in df.iterrows():
  source_x = row[f'object_{row["HMM"]}_x']
  source_y = row[f'object_{row["HMM"]}_y']
  for object_idx in range(_NUM_OBJECTS):
    if object_idx != row['HMM']:
      is_transition = (row['next_frame_HMM'] == object_idx)
      if not is_transition:
        if random.random() > subsample_proportion:
          continue
      destination_x = row[f'object_{object_idx}_x']
      destination_y = row[f'object_{object_idx}_y']
      distance = math.sqrt((source_x - destination_x)**2
                         + (source_y - destination_y)**2)
      distances_dict['distance'].append(distance)
      distances_dict['is_transition'].append(is_transition)
      distances_dict['is_from_target'].append(row['HMM'] == 0)
      distances_dict['is_to_target'].append(row['next_frame_HMM'] == 0)
df = pd.DataFrame(distances_dict)
df['is_transition_from_target'] = (df['is_transition'] & df['is_from_target'])
df['is_transition_to_target'] = (df['is_transition'] & df['is_to_target'])
df['is_transition_to_distractor'] = (df['is_transition'] & ~df['is_to_target'])

plt.figure()
sns.distplot(df['distance'][df['is_transition']], label='Transitions')
sns.distplot(df['distance'][~df['is_transition']], label='Non-Transitions')
plt.xlabel('Distance to object')
plt.ylabel('Probability')
plt.legend()

import pylab as plt
plt.figure()
def plot_loess(x, y, plt_idx):

  # Sort data by x-coordinate for plotting
  ind = np.argsort(x)
  x = x[ind]
  y = y[ind]

  l = loess(x, y, surface='direct')
  l.fit()
  pred = l.predict(x, stderror=True)
  conf = pred.confidence(alpha=0.01)
  
  lowess = pred.values
  ll = np.maximum(0, conf.lower)
  ul = np.minimum(1, conf.upper)
  
  plt.subplot(2, 2, plt_idx)
  plt.plot(x, y, '+')
  plt.plot(x, lowess)
  plt.xlim(right=1100)
  y_margin = subsample_proportion/20
  plt.ylim(bottom=-y_margin, top=subsample_proportion+y_margin)
  if plt_idx % 2 == 1:
    plt.ylabel('Transition probability')
  if plt_idx > 2:
    plt.xlabel('Distance to object')
  plt.fill_between(x,ll,ul,alpha=.33)

# Correct for the fact that we downsampled non-transitions by 1000
df['is_transition'] *= subsample_proportion
df['is_transition_from_target'] *= subsample_proportion
df['is_transition_to_target'] *= subsample_proportion
df['is_transition_to_distractor'] *= subsample_proportion

plot_loess(x=df['distance'], y=df['is_transition'], plt_idx=1)
plt.title(r'Any object $\to$ Any object')

df_from_target = df[df['is_from_target']].reset_index()
plot_loess(x=df_from_target['distance'],
           y=df_from_target['is_transition_from_target'],
           plt_idx=2)
plt.title(r'Target $\to$ Distractor')

df_from_distractor = df[~df['is_from_target']].reset_index()
plot_loess(x=df_from_distractor['distance'],
           y=df_from_distractor['is_transition_to_target'],
           plt_idx=3)
plt.title(r'Distractor $\to$ Target')

plot_loess(x=df_from_distractor['distance'],
           y=df_from_distractor['is_transition_to_distractor'],
           plt_idx=4)
plt.title(r'Distractor $\to$ Distractor')

plt.tight_layout()
plt.show()
