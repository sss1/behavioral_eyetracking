import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import naive_eyetracking
from eyetracking_hmm import performance_according_to_HMM
import pickle
np.set_printoptions(threshold = np.nan)

# This script generates all of the videos that were used by human coders to
# hand-code the eye-tracking data.
# By changing prediction_to_plot below, the same script can be used to plot
# the predictions of the HMM and/or Naive model.

cache_file = '300.pickle'

is_supervised = False # Set to True for supervised TrackIt (with changing target object)
target_color = None # Usually 'b'; None means random, for human coding
target_name = 'Distractor 0' # Usually 'Target'
distractor_color = None # Usually 'r'

save_video = False # If True, saves the output video. If False, only displays the video.
root = './' # root directory 

# boundaries of track-it grid
x_min = 400
x_max = 2000
y_min = 0
y_max = 1200

space = 50 # number of extra pixels to display on either side of the plot

prediction_to_plot = 'None' # Should be one of 'HMM', 'Naive', or 'None'

lag = 10 # plot a time window of length lag, so we can see the trajectory more clearly

experiments_to_show = [(str(subject_idx, 'shrinky')) for subject_idx in range(50)] \
                    + [(str(subject_idx, 'noshrinky')) for subject_idx in range(50)]

with open(cache_file , 'r') as f:
  subjects = pickle.load(f)

def plot_video(subject_ID, experiment):

  trackit_data = subjects[subject_ID].experiments[experiment].datatypes['trackit']
  trials_to_show = range(int(trackit_data.metadata['Trial Count']))[1:] # Omit first (practice) trial
  eyetrack_data = subjects[subject_ID].experiments[experiment].datatypes['eyetrack']
  
  print('Number of trials: ' + str(len(trials_to_show)))
  
  # Set up formatting for the saved movie file
  if save_video:
    Writer = animation.writers['ffmpeg']
    relative_speed = 0.1
    original_fps = 60
    writer = Writer(fps = relative_speed * original_fps, bitrate = 1800)
  
  for trial_idx in trials_to_show:
  
    trackit_trial, eyetrack_trial = trackit_data.trials[trial_idx], eyetrack_data.trials[trial_idx]
  
    print('Trial: ' + str(trial_idx) + '    Proportion of data missing: ' + str(eyetrack_trial.proportion_missing))
    eyetrack = eyetrack_trial.data[:, 1:]
    target = trackit_trial.object_positions[trackit_trial.target_index, :, :]
    distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0) # object X frame X coordinate
    if prediction_to_plot == 'HMM':
      performance_according_to_HMM(trackit_trial, eyetrack_trial)
      MLE = eyetrack_trial.HMM_MLE
    elif prediction_to_plot == 'Naive':
      MLE = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))
  
    trial_length = target.shape[0]
   
    # initializate plot background and objects to plot
    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
    eyetrack_line, = ax.plot([], [], c = 'k', lw = 2, label = 'Eye-track')
    distractors_lines = []
    if prediction_to_plot != 'None':
      state_point = ax.scatter([], [], s = 75, c = 'g', label = 'Model Prediction')
    frame_text = ax.text(1800, 25, str(0))
    if is_supervised:
      trackit_line, = ax.plot([], [], lw = 2, c = 'r', label = 'Object 1')
      for j in range(len(distractors)):
          distractors_lines.extend(ax.plot([], [], c = distractor_color, lw = 2, label = 'Object ' + str(j + 1)))
    else:
      trackit_line, = ax.plot([], [], lw = 2, c = target_color, label = target_name)
      for j in range(len(distractors)):
        distractors_lines.extend(ax.plot([], [], c = distractor_color, lw = 2, label = 'Distractor ' + str(j + 1)))
  
    legend_entries = [eyetrack_line, trackit_line]
    legend_entries.extend(distractors_lines)
    if prediction_to_plot != 'None':
      legend_entries.append(state_point)
    plt.legend(handles = legend_entries, loc = 'upper right')
  
    # Rather than a single point, show tail of object trajectories (frames in range(trial_length - lag))
    # This makes it much easier to follow objects visually
    def animate(i):
      if i % 100 == 0:
        print('Current frame: ' + str(i))
      frame_text.set_text(str(i))
      trackit_line.set_data(target[i:(i + lag),0], target[i:(i + lag),1])
      eyetrack_line.set_data(eyetrack[i:(i + lag),0], eyetrack[i:(i + lag),1])
      for j in range(len(distractors)):
        distractors_lines[j].set_data(distractors[j,i:(i + lag),0],
                                      distractors[j,i:(i + lag),1])
      if prediction_to_plot != 'None':
        state = MLE[i + lag]
        if state == 0:
          state_point.set_offsets(target[i + lag - 1,:])
        elif state > 0:
          state_point.set_offsets(distractors[state - 1, i + lag - 1, :])
        elif state < 0:
          state_point.set_offsets([0, 0])
      plt.draw()
      plt.xlim([x_min, x_max])
      plt.ylim([y_min, y_max])
      return trackit_line, eyetrack_line,
  
  
    anim = animation.FuncAnimation(fig, animate,
                                   frames = trial_length - lag,
                                   interval = 8.33,
                                   blit = False,
                                   repeat = False)
    if save_video:
      video_dir = root
      save_path = video_dir + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_uncoded.mp4'
      print('Saving video to ' + save_path)
      anim.save(save_path, writer = writer)
    else:
      plt.show()

print('Generating videos: ' + str(experiments_to_show))

for (subject_ID, experiment) in experiments_to_show:
  plot_video(subject_ID, experiment)
