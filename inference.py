from __future__ import print_function, division
from functions import *
import paddle
import paddle.fluid.layers as pd
import time
import logging
import random
from threading import Thread
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageFont
import os
import sys
import pickle
import matplotlib.pyplot as plt
from model import *
from test import *
from data import *
import argparse
import scipy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('resume', nargs='?', default='0')
args = parser.parse_args()

import os
#os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
gpu_num = 0
new_model = False#args.resume!='1'
batch_size = 20

print("Initialising Tensors")
alpha = pd.data(name='alpha', shape=[1], dtype='float32', append_batch_size=False)
input = pd.data(name='input', shape=[-1, 3, 384, 512], dtype='float32')
target15 = pd.data(name='target15', dtype='float32' , shape=(-1, 1, 1, 1))
target14 = pd.data(name='target14', dtype='float32' , shape=(-1, 1, 3, 4))
target13 = pd.data(name='target13', dtype='float32' , shape=(-1, 1, 6, 8))
target12 = pd.data(name='target12', dtype='float32' , shape=(-1, 1, 12, 16))
target11 = pd.data(name='target11', dtype='float32' , shape=(-1, 1, 24, 32))
target10 = pd.data(name='target10', dtype='float32' , shape=(-1, 1, 48, 64))

loss, out15, out14, out13, out12, out11, out10, test_program, monitor = model(input,
                                                                     [target15, target14, target13, target12, target11, target10], alpha)

place = paddle.fluid.CUDAPlace(0)
exe = paddle.fluid.Executor(place)
main_program = paddle.fluid.default_main_program()

best_saver = Saver(exe=exe, path='./output/best_model', max_to_keep=1)
saver = Saver(exe=exe, path='./output/model', max_to_keep=2)
#   print('total number of parameters:', total_parameters())
def get_font_size(height):
  fontsize=10
  font = ImageFont.truetype("arial.ttf", fontsize)
  while font.getsize('HELLO')[1] < height:
    # iterate until the text size is just larger than the criteria
    fontsize += 2
    font = ImageFont.truetype("arial.ttf", fontsize)
  return fontsize-1

def get_heatmap(dmap):
  cm = plt.get_cmap('jet')
  dmap = cm(dmap)
  dmap = np.uint8(dmap*255)
  return dmap

def get_data_by_names(names):

    imgs = []

    for name in names:
      if not name == 'NULL':
        img = cv2.imread(name+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
#        img = np.asarray(Image.open(name+'.jpg'))
      imgs.append(img)

    imgs = np.array([normalise(img) for img in imgs])
    return imgs

def full_test(exe, test_program, fetch_list, part='B', gpu_num=1):
  strict_test_names, test_dict = get_test_data_names()
  input_shape = (384, 512)
  dmap_shape = (48, 64)
  pbar = tqdm(total=len(strict_test_names))
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
      pbar.update(1)
  pbar.close()
  results = []
  for key in test_dict:
    if key != 'names_to_name':
      _data = test_dict[key]
      results.append(np.abs(np.round(_data['predict'])-round(_data['truth']))/max(1,round(_data['truth'])))

  results = np.mean(results, axis=0)
  return results

if True:
    best_saver.restore( best_saver.last_checkpoint() )
#    saver.restore( saver.last_checkpoint() )
    test_results = full_test(exe, test_program,
                fetch_list=[out15, out14, out13, out12, out11, out10],
                gpu_num=gpu_num)
    print(test_results)
