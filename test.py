from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import sys
import time
import pickle

from functions import *
from model import *
from data import *

def get_test_names(part='B'):
    if not (os.path.exists('./test_dict.pkl') and os.path.exists('./strict_test_names.pkl')):
        test_dict = {}
        strict_test_names = preprocess_data(
            names=load_data_names(train=False, part=part),
            data_path='./datasets/shanghaitech/'+part+'/test/',
            test=True,
            test_dict=test_dict,
            input_size=[384,512],
            load_data_fn=load_data_ShanghaiTech
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
    return strict_test_names, test_dict

def get_data_by_names(names):

    imgs = []
    targets15 = []
    targets14 = []
    targets13 = []
    targets12 = []
    targets11 = []
    targets10 = []

    for name in names:
      if not name == 'NULL':
          img = np.asarray(Image.open(name+'.jpg'))
          target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
      imgs.append(img)
      targets15.append(np.reshape(np.sum(target14), [1,1,1]))
      targets14.append(target14)
      targets13.append(target13)
      targets12.append(target12)
      targets11.append(target11)
      targets10.append(target10)

    targets = [targets15, targets14, targets13, targets12, targets11, targets10]
    return np.array(normalize(imgs)), targets

def full_test(sess, Decoded,
    input, target15, target14, target13, target12, target11, target10, training,
    part='B', gpu_num=1):
  print(">>> Test begins", end='.')
  strict_test_names, test_dict = get_test_names(part)
  for key in test_dict:
    test_dict[key]['predict'] = np.array([0.0]*6)

  step = 0
  batch_names = []
  for k in range(len(strict_test_names)):
    test_name_strict = strict_test_names[k]

    batch_names.append(test_name_strict)
    if len(batch_names) < gpu_num and k < len(strict_test_names)-1:
        continue
    else:
        if k == len(strict_test_names)-1:
            while len(batch_names) < gpu_num:
                batch_names.append('NULL')
        test_inputs, test_targets = get_data_by_names(batch_names)
        test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
        test_D  = sess.run(Decoded, feed_dict={
            input: test_inputs,
            target15: test_t15,
            target14: test_t14,
            target13: test_t13,
            target12: test_t12,
            target11: test_t11,
            target10: test_t10,
            training: False,
        })
        out15, out14, out13, out12, out11, out10 = test_D
        for i in range(len(batch_names)):
            name = batch_names[i]
            if not name == 'NULL':
                data_name = test_dict['names_to_name'][name]
                test_dict[data_name]['predict'] += np.array([np.sum(out15[i]),np.sum(out14[i])
                        ,np.sum(out13[i]),np.sum(out12[i]),np.sum(out11[i]),np.sum(out10[i])])

        batch_names = []

    step += 1

    if step % (len(strict_test_names)//15) == 0:
        print('.', end='')
        sys.stdout.flush()
  print()

  results = []
  for key in test_dict:
    if key != 'names_to_name':
      _data = test_dict[key]
      results.append(np.abs(_data['predict']-_data['truth']))

  results = np.mean(results, axis=0)
  return results
