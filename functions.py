from __future__ import print_function, division
import torch
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import string
import os
import random
import time
from tqdm import tqdm
import math
import pickle

def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def total_parameters(scope=None):
  total_parameters = 0
  for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.value
      total_parameters += variable_parameters
  return total_parameters
def gaussian_kernel(shape=(32,32),sigma=5):
  radius_x,radius_y = [(radius-1.)/2. for radius in shape]
  y_range,x_range = np.ogrid[-radius_y:radius_y+1,-radius_x:radius_x+1]
  h = np.exp( -(x_range*x_range + y_range*y_range) / (2.*sigma*sigma) )

  h[ h < np.finfo(h.dtype).eps*h.max()] = 0
  sumofh = h.sum()
  if sumofh != 0:
      h /= sumofh
  return h
def get_downsized_density_maps(density_map):
  ddmaps = []
  ratios = [8,16,32,64,128]
  with tf.device('/gpu:0'):
    ddmap = tf.layers.average_pooling2d(density_map, ratios[0], ratios[0], padding='same') * (ratios[0] * ratios[0])
    ddmaps.append(tf.squeeze(ddmap,0))
    if len(ratios)>1:
      for i in range(len(ratios)-1):
        ratio = int(ratios[i+1]/ratios[i])
        ddmap = tf.layers.average_pooling2d(ddmap, ratio, ratio, padding='same') * (ratio * ratio)
        ddmaps.append(tf.squeeze(ddmap,0))
  return ddmaps, [tf.image.flip_left_right(ddmap) for ddmap in ddmaps]
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(round(img_width/input_width)))
  rows = max(1, int(round(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def get_coords_map(coords, resize, img_size):
  resized_height, resized_width = resize
  img_height, img_width = img_size
  new_coords = []
  for coord in coords:
    new_coord = [0,0]
    new_coord[0] = min(coord[0], img_width-1)*resized_width/img_width
    new_coord[1] = min(coord[1], img_height-1)*resized_height/img_height
    new_coords.append(new_coord)
  coords_map = np.zeros([1, resized_height, resized_width, 1])
  for coord in new_coords:
    coords_map[0][int(coord[1])][int(coord[0])][0] += 1
  return coords_map

def preprocess_data(names, data_path, save_path='./processed', random_crop=None, divide=True, input_size=[384, 512]
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
  kernel_size = 49
  kernel = gaussian_kernel(shape=(kernel_size,kernel_size),sigma=10)
  kernel = np.reshape(kernel, kernel.shape+(1,1))

  graph_get_dmap = tf.Graph()
  with graph_get_dmap.as_default():

    kernel = tf.constant(kernel, dtype=tf.float32)

    tf_coords_map_p = tf.placeholder(tf.float32, [1,None,None,1])
    tf_dmap = tf.nn.conv2d(tf_coords_map_p, kernel, strides=(1,1,1,1), padding='SAME')

  graph_get_downsized_dmaps = tf.Graph()
  with graph_get_downsized_dmaps.as_default():

    tf_dmap_p = tf.placeholder(tf.float32, [1,input_height,input_width,1])
    tf_ddmaps = get_downsized_density_maps(tf_dmap_p)

  sess_get_dmap = tf.Session(graph=graph_get_dmap)
  sess_get_downsized_dmaps = tf.Session(graph=graph_get_downsized_dmaps)

  for ni in tqdm(range(len(names))):
    name = data_path +  names[ni]

    img, coords = load_data_fn(name)

    if img.mode !='RGB':
      img = img.convert('RGB')
    img_width, img_height = img.size

    imgs = []
    dmaps = []

    rows, columns = fit_grid(img_height, img_width, input_size=[input_height, input_width])

    resized_height = rows*input_height
    resized_width = columns*input_width
    new_img = img.resize((resized_width, resized_height))
    coords_map = get_coords_map(coords, resize=[resized_height, resized_width], img_size=[img_height, img_width])
    dmap = sess_get_dmap.run(tf_dmap, feed_dict={
        tf_coords_map_p: coords_map
    })
    if divide:
        for row in range(rows):
          for col in range(columns):
            crop_top = input_height*row
            crop_left = input_width*col
            crop_bottom = crop_top + input_height
            crop_right = crop_left + input_width
            img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

            ddmaps, ddmaps_mirrored = sess_get_downsized_dmaps.run(tf_ddmaps, feed_dict={
                tf_dmap_p: dmap[:, crop_top:crop_bottom, crop_left:crop_right]
            })

            imgs.append(img_crop)
            dmaps.append(ddmaps)
            if not test:
              imgs.append(ImageOps.mirror(img_crop))
              dmaps.append(ddmaps_mirrored)

    if random_crop is not None and not (rows==1 and columns==1) and not test:
      for b in range(random_crop):

        crop_top = 0 if rows==1 else np.random.randint(0, resized_height - input_height)
        crop_left = 0 if columns==1 else np.random.randint(0, resized_width - input_width)
        crop_bottom = crop_top + input_height
        crop_right = crop_left + input_width
        img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        ddmaps, ddmaps_mirrored = sess_get_downsized_dmaps.run(tf_ddmaps, feed_dict={
            tf_dmap_p: dmap[:, crop_top:crop_bottom, crop_left:crop_right]
        })

        imgs.append(img_crop)
        dmaps.append(ddmaps)

        imgs.append(ImageOps.mirror(img_crop))
        dmaps.append(ddmaps_mirrored)

    for i in range(len(imgs)):
      new_name = id_generator()

      img_i = imgs[i]
      if not test and random.random()>0.9:
        img_i = img_i.convert('L').convert('RGB')
      img_i.save(save_path + new_name + '.jpg', 'JPEG')
      with open(save_path + new_name + '.pkl', 'wb') as f:
        pickle.dump(dmaps[i], f)

      out_names.append(save_path + new_name)

      if test:
        test_dict['names_to_name'][save_path + new_name] = name
    if test:
      test_dict[name] = {
          'predict': -1,
          'truth': np.sum(dmap)
      }
  return out_names

def set_pretrained(sess):

  torch_dict = torch.load('vgg16-397923af.pth')

  tf_p_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  torch_p_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]
  trainables = tf.trainable_variables()
  for i in range(10):
    tf_name_w = 'vgg_conv_'+str(tf_p_ids[i])+'/kernel:0'
    tf_name_b = 'vgg_conv_'+str(tf_p_ids[i])+'/bias:0'

    torch_name_w = 'features.'+str(torch_p_ids[i])+'.weight'
    torch_name_b = 'features.'+str(torch_p_ids[i])+'.bias'

    var_w = [v for v in trainables if v.name == tf_name_w ][0]
    sess.run(tf.assign(var_w, np.transpose(torch_dict[torch_name_w].data.numpy(), (2,3,1,0))))

    var_b = [v for v in trainables if v.name == tf_name_b ][0]
    sess.run(tf.assign(var_b, torch_dict[torch_name_b].data.numpy()))
#   test_set_pretrained('CAC/vgg_conv_10/kernel:0', 'features.21.weight', torch_dict)
def test_set_pretrained(tf_name, torch_name, torch_dict):
  def check_equal(a, b):
    a = a.flatten()
    b = b.flatten()
    if len(a) != len(b):
        print('inequivalent length:', len(a), '!=', len(b))
        return False
    for m in range(len(a)):
        if abs(a[m]-b[m]) > 0.000001:
            print(a[m], '!=', b[m], 'at', m)
            return False
    return True
  tf_data = [v for v in tf.trainable_variables() if v.name ==tf_name][0].read_value().eval()
  torch_data = torch_dict[torch_name].data.numpy()
  assert check_equal(tf_data, torch_data)
def moving_average(new_val, last_avg, theta=0.95):
  return round((1-theta) * new_val + theta* last_avg, 2)
def moving_average_array(new_vals, last_avgs, theta=0.95):
  return [round((1-theta) * new_vals[i] + theta* last_avgs[i], 2) for i in range(len(new_vals))]
def MAE(predicts, targets):
  return round( np.mean( np.absolute( np.sum(predicts, (1,2,3)) - np.sum(targets, (1,2,3)) )), 1)
def normalize(imgs):
  new_imgs = []
  for i in range(len(imgs)):
    img = imgs[i] / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    new_imgs.append(img)
  return new_imgs
def denormalize(img):
  img *= [0.229, 0.224, 0.225]
  img += [0.485, 0.456, 0.406]
  img *= 255
  return img.astype('uint8')
def next_batch(batch_size, names):
  b = np.random.randint(0, len(names), [batch_size])
  _names = names[b]

  imgs = []
  targets15 = []
  targets14 = []
  targets13 = []
  targets12 = []
  targets11 = []
  targets10 = []

  for name in _names:
    imgs.append(np.asarray(Image.open(name+'.jpg')))
    target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
    targets15.append(np.reshape(np.sum(target14), [1,1,1]))
    targets14.append(target14)
    targets13.append(target13)
    targets12.append(target12)
    targets11.append(target11)
    targets10.append(target10)

  targets = [targets15, targets14, targets13, targets12, targets11, targets10]
  return np.array(normalize(imgs)), targets
