from __future__ import print_function, division
import numpy as np
import paddle
import paddle.fluid
from PIL import Image
import string
import os
import random
import time
#from tqdm import tqdm
import math
import pickle

def display_set_of_imgs(images, rows=2, size=0.5, name='0'):
  n_images = len(images)
  with open('./output/images/'+str(name)+'-'+id_generator(5)+'.pkl', 'wb') as f:
      pickle.dump(images, f)
def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def total_parameters(scope=None):
  total_parameters = 0
  for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.value
      total_parameters += variable_parameters
  return total_parameters

def set_pretrained(place):
  with open('vgg16.pkl','rb') as f: 
    torch_dict = pickle.load(f)
  scope = paddle.fluid.global_scope()
  
  def check_equal(pd_name, torch_name):
    a = np.array(scope.find_var(pd_name).get_tensor())
    b = np.array(torch_dict[torch_name])
    a = a.flatten()
    b = b.flatten()
    assert len(a) == len(b), 'inequivalent length'
    for m in range(len(a)):
        assert abs(a[m]-b[m]) < 0.000001, 'difference found at '+str(m)

  paddle_p_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  torch_p_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]
  for i in range(10):
    paddle_name_w = 'vgg_conv_'+str(paddle_p_ids[i])+'.w_0'
    paddle_name_b = 'vgg_conv_'+str(paddle_p_ids[i])+'.b_0'

    torch_name_w = 'features.'+str(torch_p_ids[i])+'.weight'
    torch_name_b = 'features.'+str(torch_p_ids[i])+'.bias'

    var = scope.find_var(paddle_name_w).get_tensor()
    var.set(torch_dict[torch_name_w], place)
    
    var = scope.find_var(paddle_name_b).get_tensor()
    var.set(torch_dict[torch_name_b], place)
#     check_equal(paddle_name_w, torch_name_w)
#     check_equal(paddle_name_b, torch_name_b)

def moving_average(new_val, last_avg, theta=0.95):
  return round((1-theta) * new_val + theta* last_avg, 2)
def moving_average_array(new_vals, last_avgs, theta=0.95):
  return [round((1-theta) * new_vals[i] + theta* last_avgs[i], 2) for i in range(len(new_vals))]
def MAE(predicts, targets):
  return round( np.mean( np.absolute( np.sum(predicts, (1,2,3)) - np.sum(targets, (1,2,3)) )), 1)
