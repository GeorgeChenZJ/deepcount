import numpy as np
from PIL import Image, ImageOps
import string
import os
import random
import time
from tqdm import tqdm
import math
import pickle
import json
import scipy.io as scipy_io


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(math.ceil(img_width/input_width)))
  rows = max(1, int(math.ceil(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def preprocess_data(names, data_path, save_path='./processed', random_crop=None, divide=True, input_size=[384, 512], annotations=None
                    , test=False, test_dict=None, load_data_fn=None):
  assert load_data_fn is not None and hasattr(load_data_fn, '__call__'), 'a function for loading image and coordinates must be given'
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  if test and not 'names_to_name' in test_dict:
    test_dict['names_to_name'] = {}

  input_height, input_width = input_size
  prog = 0
  out_names = []

  for ni in tqdm(range(len(names))):
    name = data_path +  names[ni]

    img, num = load_data_fn(name, annotations)

    if img.mode !='RGB':
      img = img.convert('RGB')
    img_width, img_height = img.size

    imgs = []

    rows, columns = fit_grid(img_height, img_width, input_size=[input_height, input_width])

    resized_height = rows*input_height
    resized_width = columns*input_width
    new_img = img.resize((resized_width, resized_height))
    if divide:
        for row in range(rows):
          for col in range(columns):
            crop_top = input_height*row
            crop_left = input_width*col
            crop_bottom = crop_top + input_height
            crop_right = crop_left + input_width
            img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

            imgs.append(img_crop)

    for i in range(len(imgs)):
      new_name = id_generator()

      img_i = imgs[i]
      img_i.save(save_path + new_name + '.jpg', 'JPEG')
      out_names.append(save_path + new_name)
      if test:
        test_dict['names_to_name'][save_path + new_name] = name
    if test:
      test_dict[name] = {
          'predict': -1,
          'truth': num
      }
  return out_names
def save_test_data_names():
    if not (os.path.exists('./test_dict.pkl') and os.path.exists('./strict_test_names.pkl')):
        test_dict = {}
        with open('./test_data/test.json','rb') as f:
          annotations = json.load(f)
        strict_test_names = preprocess_data(
            names=load_data_names_test(),
            data_path='test_data/images/',
            test=True,
            test_dict=test_dict,
            input_size=[384,512],
            annotations=annotations,
            load_data_fn=load_data_test
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
def load_data_names_test():
  names = []
  for _, _, files in os.walk('./test_data/images/'):
    for filename in files:
      if '.jpg' in filename or '.png' in filename:
        names.append(filename)
  return names
def load_data_test(path, annotations):
  img = Image.open(path)
  name = path[path.rfind('/')+1:]
  num = annotations[name]
  return img, num

if __name__ == '__main__':
    save_test_data_names()
