from __future__ import print_function, division
import tensorflow as tf
import os
import random
import pickle
from scipy import io as scipy_io
import numpy as np

from PIL import Image

from functions import preprocess_data

data_dir = "./shanghaitech/"

def get_train_data_names(part):
    if not (os.path.exists('./train_names.pkl') and os.path.exists('./test_names.pkl')):
        move_files('./datasets/shanghaitech/'+part+'/', part = part)
        tf.reset_default_graph()
        train_names = preprocess_data(
            names=load_data_names(train=True, part= part),
            data_path='./datasets/shanghaitech/'+part+'/train/',
            random_crop=30,
            input_size=[384,512],
            load_data_fn=load_data_ShanghaiTech
        )
        random.shuffle(train_names)
        print()
        print(len(train_names), 'of training data')

        test_names = preprocess_data(
            names=load_data_names(train=False, part= part),
            data_path='./datasets/shanghaitech/'+part+'/test/',
            random_crop=5,
            input_size=[384,512],
            load_data_fn=load_data_ShanghaiTech
        )
        random.shuffle(test_names)
        print()
        print(len(test_names), 'of testing data')
        with open('train_names.pkl', 'wb') as f:
            pickle.dump(train_names, f)
        with open('test_names.pkl', 'wb') as f:
            pickle.dump(test_names, f)
    else:
        train_names = pickle.load(open('./train_names.pkl', 'rb'))
        test_names = pickle.load(open('./test_names.pkl', 'rb'))
    return np.array(train_names), np.array(test_names)
def move_files(path_to_load, part='A'):
  if not path_to_load.endswith('/'):
    path_to_load += '/'
  train_ptl = path_to_load + 'train/'
  test_ptl = path_to_load + 'test/'

  if not os.path.exists(train_ptl):
    os.makedirs(train_ptl)
  if not os.path.exists(test_ptl):
    os.makedirs(test_ptl)
  for _, _, files in os.walk(data_dir+"/part_"+part+"_final/train_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename(data_dir+"/part_"+part+"_final/train_data/ground_truth/"+filename, train_ptl + new_name)
        os.rename(data_dir+"/part_"+part+"_final/train_data/images/"+new_name.replace('.mat','.jpg'), train_ptl + new_name.replace('.mat','.jpg'))
  for _, _, files in os.walk(data_dir+"/part_"+part+"_final/test_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename(data_dir+"/part_"+part+"_final/test_data/ground_truth/"+filename, test_ptl + new_name)
        os.rename(data_dir+"/part_"+part+"_final/test_data/images/"+new_name.replace('.mat','.jpg'), test_ptl + new_name.replace('.mat','.jpg'))

def load_data_names(train=True, part='A'):
  names = []
  if train:
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/train/'):
      for filename in files:
        if '.mat' in filename:
          names.append(filename.replace('.mat',''))
  else:
    pass
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/test'):
        for filename in files:
          if '.jpg' in filename:
            names.append(filename.replace('.jpg',''))
  return names
def load_data_ShanghaiTech(path):
  img = Image.open(path+'.jpg')
  coords = scipy_io.loadmat(path+'.mat')['image_info'][0][0][0][0][0]
  return img, coords
