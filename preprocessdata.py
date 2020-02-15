import numpy as np
from PIL import Image, ImageOps
import string
import os
import random
import time
#from tqdm import tqdm
import math
import pickle
import json
import scipy.io as scipy_io
import shutil


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(round(img_width/input_width)))
  rows = max(1, int(round(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def get_resize_ratio(size):
  threshold = 500
  min_ratio = 0.5
  if size > threshold:
    return ((size-threshold) * min_ratio + threshold) / size
  return 1.0
def preprocess_data(names, data_path, save_path, input_size=[384, 512], annotations=None):
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  input_height, input_width = input_size
  prog = 0
  out_names = []

  #for ni in tqdm(range(len(names))):
  for ni in range(len(names)):
    name = names[ni]
    if annotations is not None:
      coords = annotations[name]
    else:
      coords = []

    new_name = save_path + id_generator(size=7)

    shutil.copyfile(data_path + name, new_name+'.jpg')

    with open(new_name+'.pkl','wb') as f:
      pickle.dump(coords, f)
    out_names.append(new_name)
  return out_names

def preprocess_test_data(names, data_path, save_path, input_size=[384, 512], annotations=None, test_dict=None, load_data_fn=None):
  assert load_data_fn is not None and hasattr(load_data_fn, '__call__'), 'a function for loading image and coordinates must be given'
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  if not 'names_to_name' in test_dict:
    test_dict['names_to_name'] = {}

  input_height, input_width = input_size
  prog = 0
  out_names = []

  #for ni in tqdm(range(len(names))):
  for ni in range(len(names)):
    name = data_path +  names[ni]

    img, num = load_data_fn(name, annotations)

    if img.mode !='RGB':
      img = img.convert('RGB')
    img_width, img_height = img.size

    imgs = []

    ratio = get_resize_ratio( max(img_height, img_width) )
    rows, columns = fit_grid(img_height*ratio, img_width*ratio, input_size=[input_height, input_width])

    resized_height = rows*input_height
    resized_width = columns*input_width
    new_img = img.resize((resized_width, resized_height))
    for row in range(rows):
      for col in range(columns):
        crop_top = input_height*row
        crop_left = input_width*col
        crop_bottom = crop_top + input_height
        crop_right = crop_left + input_width
        img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        imgs.append(img_crop)

    for i in range(len(imgs)):
      new_name = save_path + id_generator(size=6)
      imgs[i].save(new_name + '.jpg', 'JPEG', quality=90)
      out_names.append(new_name)
      test_dict['names_to_name'][new_name] = name
    test_dict[name] = {
        'predict': -1,
        'truth': num
    }
  return out_names
def save_train_data_names():
    if not (os.path.exists('./train_names.pkl') and os.path.exists('./test_names.pkl')):
        with open('./data/train.json','rb') as f:
          annotations = json.load(f)
        all_names = annotations.keys()
        random.shuffle(all_names)
        train_names = preprocess_data(
            names=all_names,
            data_path='./data/train/',
            save_path='pr/',
            annotations=annotations,
        )
        random.shuffle(train_names)
        print()
        print(len(train_names), 'of training data')
        with open('./data/test.json','rb') as f:
          annotations = json.load(f)
        all_names = annotations.keys()
        test_names = preprocess_data(
            names=all_names,
            data_path='./data/test/',
            save_path='prt/',
            annotations=annotations,
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
        print('found train_names.pkl with data', len(train_names), 'test_names.pkl with data', len(test_names))
def save_test_data_names():
    if not (os.path.exists('./test_dict.pkl') and os.path.exists('./strict_test_names.pkl')):
        test_dict = {}
        with open('./data/test_strict.json','rb') as f:
          annotations = json.load(f)
        strict_test_names = preprocess_test_data(
            names=annotations.keys(),
            data_path='./data/test/',
            save_path='prtest/',
            test_dict=test_dict,
            annotations=annotations,
            load_data_fn=load_data_cc_data_valid
        )
        random.shuffle(strict_test_names)
        print()
        print(len(strict_test_names), 'of data')
        with open('strict_test_names.pkl', 'wb') as f:
            pickle.dump(strict_test_names, f)
        with open('test_dict.pkl', 'wb') as f:
            pickle.dump(test_dict, f)
    else:
        strict_test_names = pickle.load(open('./strict_test_names.pkl', 'rb'))
        test_dict = pickle.load(open('./test_dict.pkl', 'rb'))
        print('found strict_test_names.pkl with data', len(strict_test_names))
def load_data_names_cc_data_valid():
  names = []
  for _, _, files in os.walk('./cc_data_valid/data/'):
    for filename in files:
      names.append(filename)
  return names
def load_data_cc_data_valid(path, annotations):
  img = Image.open(path)
  name = path[path.rfind('/')+1:]
  num = annotations[name]
  return img, num
if __name__ == '__main__':
    save_train_data_names()
    save_test_data_names()
