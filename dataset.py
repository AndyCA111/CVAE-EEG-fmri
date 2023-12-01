from torch.utils.data import Dataset
import math
import numpy as np
import nibabel.processing
from nilearn import image, masking
import nibabel as nib
import mne
import json
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.transform import resize

# mne.utils.use_log_level('error')
mne.use_log_level(False)

def get_eeg_subjects(dir="/content/drive/MyDrive/EEG_GAN/datasets/datasets/ds002338"):
    # eeg files are inside 'derivatives'
    eeg_root = os.path.join(dir, 'derivatives')
    fmri_root = dir
    eeg_subjects = [f for f in os.listdir(eeg_root) if 'sub' in f]
    # print(eeg_subjects)
    _1dNF_paths = []
    _2dNF_paths = []
    _pre_paths = []
    _post_paths = []
    eeg_fmri_table = {}
    for sub in eeg_subjects:
      path = os.path.join(eeg_root, sub, 'eeg_pp')
      experiment_files = [f for f in os.listdir(path) if 'vhdr' in f]
      # print(experiment_files)

      fmri_path = os.path.join(fmri_root, sub, 'func')

      for exp in experiment_files:
        eeg_p = os.path.join(path, exp)
        if '1dNF' in exp:
          _1dNF_paths.append(eeg_p)
          if 'run-01' in exp:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-1dNF_run-01_bold.nii.gz')
          elif 'run-02' in exp:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-1dNF_run-02_bold.nii.gz')
          else:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-1dNF_run-03_bold.nii.gz')
        elif '2dNF' in exp:
          _2dNF_paths.append(eeg_p)
          if 'run-01' in exp:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-2dNF_run-01_bold.nii.gz')
          elif 'run-02' in exp:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-2dNF_run-02_bold.nii.gz')
          else:
            fmri_p = os.path.join(fmri_path, f'{sub}_task-2dNF_run-03_bold.nii.gz')
        else: # control
          if 'post' in exp:
            _post_paths.append(eeg_p)
            fmri_p = os.path.join(fmri_path, f'{sub}_task-MIpost_bold.nii.gz')
          else:
            _pre_paths.append(eeg_p)
            fmri_p = os.path.join(fmri_path, f'{sub}_task-MIpre_bold.nii.gz')

        eeg_fmri_table[eeg_p] = fmri_p

    return _1dNF_paths, _2dNF_paths, _pre_paths, _post_paths, eeg_fmri_table

def parse_one_file(eeg_signal, fmri, start_time, end_time, pre_post, window=4, sampling_rate=200, lag=0, resize=True):
    result_eeg = []
    result_fmri = []
    step = window*sampling_rate
    scaler = MinMaxScaler()
    if pre_post:
      step = sampling_rate # shorter step due to insufficient data

    for t in range(start_time, end_time, step):
        s = eeg_signal[:, t:t+sampling_rate] # take 1 second of eeg data, considering the potential size of model
        assert s.shape == (64, 200)
        # eeg normalization??
        mini = np.min(s, axis=1, keepdims=True)
        maxi = np.max(s, axis=1, keepdims=True)
        normed = (s - mini) / (maxi - mini)
        result_eeg.append(normed)

        fmri_slice_start = math.ceil(start_time/sampling_rate)
        sub_volume = fmri[:, :, :, lag+fmri_slice_start]
        # per slice fmri standardization
        mini = np.min(sub_volume, axis=(0, 1), keepdims=True)
        maxi = np.max(sub_volume, axis=(0, 1), keepdims=True)
        f = (sub_volume - mini) / (maxi - mini)
        if resize:
            f = resize(f, (64, 64, 16), mode='constant', cval=0)
        # print(avg.shape)
#         assert avg.shape == fmri.shape[:3]
        result_fmri.append(f)
    # print("Finished parsing")
    # print(np.array(result_eeg).shape)
    # print(np.array(result_fmri).shape)
    print("Finished parsing one file")
    return result_eeg, result_fmri

def extract_eeg_fmri_pairs_one_file(raw_eeg, raw_fmri, target_code=2, rest_code=99, pre_post=False, ratio = 0.9):
    events, annotations = mne.events_from_annotations(raw_eeg)
    signal = raw_eeg.get_data()
    all_eeg = []
    all_fmri = []
    start = None
    end = None
    find_start = False
    start_end_times = []
    for time, _, event_code in events:
        if event_code == target_code:
            # this is a task trial
            start = time
            find_start = True
        elif event_code == rest_code and find_start:
            # print("Find the end of a stimulus event: ")
            # print(f"Start: {start}; End: {time}")
            start_end_times.append(str([start, time]))
            eeg_fragments, fmri_fragments = parse_one_file(signal, raw_fmri, start, time, pre_post=pre_post)
            all_eeg.extend(eeg_fragments)
            all_fmri.extend(fmri_fragments)
            find_start = False

    all_eeg = np.array(all_eeg)
    all_fmri = np.array(all_fmri)
    train_eeg = all_eeg[:int(all_eeg.shape[0]*ratio)]
    test_eeg = all_eeg[int(all_eeg.shape[0]*ratio):]
    train_fmri = all_fmri[:int(all_fmri.shape[0]*ratio)]
    test_fmri = all_fmri[int(all_fmri.shape[0]*ratio):]
    return train_eeg, train_fmri, test_eeg, test_fmri, start_end_times


def segment_data(dataset_table):
    all_subject_eeg_train = []
    all_subject_eeg_test = []
    all_subject_fmri_train = []
    all_subject_fmri_test = []
    all_subject_label_train = []
    all_subject_label_test = []
    inf = {}
    for eeg_file in dataset_table.keys():
      print(f"=====>>>> Segmenting {eeg_file}")
      eeg = mne.io.read_raw_brainvision(eeg_file)
      events, annotations = mne.events_from_annotations(eeg)
      # print(events[:10])
      try:
        fmri_img = image.load_img(dataset_table[eeg_file])
        fmri_img = image.clean_img(fmri_img, t_r=7, standardize=False)
        fmri_img = image.smooth_img(fmri_img, 3.0)
        fmri = fmri_img.get_fdata()
      except:
        print(f"Failed to load {dataset_table[eeg_file]}... Skipping it")
        continue
      if '1dNF' in eeg_file:
        label = 0
      elif '2dNF' in eeg_file:
        label = 1
      elif 'pre' in eeg_file:
        label = 2
      else:
        label = 3
      eeg_seg_train, fmri_seg_train, eeg_seg_test, fmri_seg_test, times = extract_eeg_fmri_pairs_one_file(eeg, fmri) # one subject
      inf[eeg_file] = times
      all_subject_eeg_train.extend(eeg_seg_train)
      all_subject_eeg_test.extend(eeg_seg_test)

      all_subject_fmri_train.extend(fmri_seg_train)
      all_subject_fmri_test.extend(fmri_seg_test)
      all_subject_label_train.extend([label]*len(eeg_seg_train))
      all_subject_label_test.extend([label]*len(eeg_seg_test))
    with open('times.json', 'w') as fr:
      json.dump(inf, fr, indent=4)
    return all_subject_eeg_train, all_subject_eeg_test, all_subject_fmri_train, all_subject_fmri_test, all_subject_label_train, all_subject_label_test


# class EEG_fMRI_Dataset(Dataset):
#   def __init__(self, root_dir, eeg_only=True, preprocessed=True, window=4, sampling_rate=200, lag=0, resample=False):
#     self.eeg = eeg_only
#     self.window = window
#     self.sampling_rate = sampling_rate
#     self.lag = lag
#     self.resample = resample
#     self.info = {}

#     with open('eeg_fmri_table.json', 'w') as fr:
#       json.dump(self.dataset_table, fr, indent=4)
#     # self.info['1dNF_samples'] = len(self._1dNF)
#     # self.info['2dNF_samples'] = len(self._2dNF)
#     # self.info['MIpre_samples'] = len(self.pre)
#     # self.info['MIpost_samples'] = len(self.post)
#     # self.skipped = []
#     self.eeg_data, self.fmri_data, self.label_data = self.segment_data()



#   def segment_data(self):
#     all_subject_eeg = []
#     all_subject_fmri = []
#     all_labels = []
#     inf = {}
#     for eeg_file in self.dataset_table.keys():
#       print(f"Segmenting {eeg_file}")
#       eeg = mne.io.read_raw_brainvision(eeg_file)
#       events, annotations = mne.events_from_annotations(eeg)
#       # print(events[:10])
#       try:
#         fmri = nib.load(self.dataset_table[eeg_file]).get_fdata()
#       except:
#         print(f"Failed to load {self.dataset_table[eeg_file]}... Skipping it")
#         self.skipped.append(eeg_file)
#         continue
#       if '1dNF' in eeg_file:
#         label = 0
#       elif '2dNF' in eeg_file:
#         label = 1
#       elif 'pre' in eeg_file:
#         label = 2
#       else:
#         label = 3
#       eeg_seg, fmri_seg, times = self.extract_eeg_fmri_pairs_one_file(eeg, fmri) # one subject
#       inf[eeg_file] = times
#       # all_subject_eeg.extend(eeg_seg)
#       # all_subject_fmri.extend(fmri_seg)
#       # all_labels.extend([label]*len(eeg_seg))
#     with open('times.json', 'w') as fr:
#       json.dump(inf, fr, indent=4)
#     return all_subject_eeg, all_subject_fmri, all_labels


#   def parse_one_file(self, eeg_signal, fmri, start_time, end_time, pre_post):
#     result_eeg = []
#     result_fmri = []
#     step = self.window*self.sampling_rate
#     if pre_post:
#       step = self.sampling_rate # shorter step due to insufficient data

#     for t in range(start_time, end_time, step):
#         s = eeg_signal[:, t:t+self.sampling_rate] # take 1 second of eeg data, considering the potential size of model
#         result_eeg.append(s)
#         assert s.shape == (64, 200)
#         fmri_slice_start = math.ceil(start_time/self.sampling_rate)
#         f = fmri[:, :, :, self.lag+fmri_slice_start] # merged the 4 frame by taking max values
#         # if self.resample:
#         #     avg = nib.processing.conform(avg, (64, 64, 16)) # TODO
#         # print(avg.shape)
# #         assert avg.shape == fmri.shape[:3]
#         result_fmri.append(f)
#     # print("Finished parsing")
#     # print(np.array(result_eeg).shape)
#     # print(np.array(result_fmri).shape)
#     return result_eeg, result_fmri


#   def __getitem__(self, index):
#     pass

#   def __len__(self):
#     pass

#   def get_summary(self):
#     return self.info


if __name__ == '__main__':
  dir = 'TODO'
  ratio = 0.8
  _1dNF, _2dNF, pre, post, dataset_table = get_eeg_subjects(dir)
  # train test split LOSO?
  train_e, test_e, train_f, test_f, train_l, test_l = segment_data(dataset_table, ratio=ratio)
  info = {}
  with open('eeg_fmri_file_matching_table.json', 'w') as fr:
    json.dump(dataset_table, fr, indent=4)
  info['1dNF_samples'] = len(_1dNF)
  info['2dNF_samples'] = len(_2dNF)
  info['MIpre_samples'] = len(pre)
  info['MIpost_samples'] = len(post)