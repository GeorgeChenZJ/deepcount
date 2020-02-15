from __future__ import print_function, division
import numpy as np
from PIL import Image, ImageOps
import os
import random
import sys
import time
import pickle

from functions import *
from model import *
from data import *

def get_data_by_names(names):

    imgs = []

    for name in names:
      if not name == 'NULL':
        img = Image.open(name+'.jpg')
        img = np.asarray(img)
      imgs.append(img)

    imgs = np.array([normalise(img) for img in imgs])
    return imgs

def full_test(exe, test_program, fetch_list, gpu_num=1):
  strict_test_names, test_dict = get_test_data_names()
  test_t15 = np.zeros([1,1,1,1]).astype('float32')
  test_t14 = np.zeros([1,1,3,4]).astype('float32')
  test_t13 = np.zeros([1,1,6,8]).astype('float32')
  test_t12 = np.zeros([1,1,12,16]).astype('float32')
  test_t11 = np.zeros([1,1,24,32]).astype('float32')
  test_t10 = np.zeros([1,1,48,64]).astype('float32')
  for key in test_dict:
    if not '.jpg' in key:
      continue
    test_dict[key]['predict'] = np.array([0.0]*6)

  for name in strict_test_names:
      test_inputs = get_data_by_names([name])
      out15, out14, out13, out12, out11, out10 = exe.run(
        program=test_program,
        fetch_list=fetch_list,
        feed={
          'input': test_inputs,
          'target15': test_t15,
          'target14': test_t14,
          'target13': test_t13,
          'target12': test_t12,
          'target11': test_t11,
          'target10': test_t10,
          'alpha': 0.0
        })
      key = test_dict['names_to_name'][name]
      test_dict[key]['predict'] += np.array([np.sum(out15[0]),np.sum(out14[0])
                        ,np.sum(out13[0]),np.sum(out12[0]),np.sum(out11[0]),np.sum(out10[0])])
  results = []
  for key in test_dict:
    if key != 'names_to_name':
      _data = test_dict[key]
      results.append(np.abs(np.round(_data['predict'])-round(_data['truth']))/max(1,round(_data['truth'])))

  results = np.mean(results, axis=0)
  return results

if __name__ == "__main__":
    test()

