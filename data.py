from __future__ import print_function, division
import pickle
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import math
from threading import Thread, Lock

def get_train_data_names():
    train_names = pickle.load(open('./train_names.pkl', 'rb'))
    test_names = pickle.load(open('./test_names.pkl', 'rb'))
    return np.array(train_names), np.array(test_names)
def get_test_data_names():
    strict_test_names = pickle.load(open('./strict_test_names.pkl', 'rb'))
    test_dict = pickle.load(open('./test_dict.pkl', 'rb'))
    return np.array(strict_test_names), test_dict

def get_downsized_density_maps(density_map):
  ddmaps = []
  ratios = [8,16,32,64,128]
  ddmap = sum_pool(density_map, 8)  
  ddmaps.append(ddmap)
  for i in range(len(ratios)-1):
    ratio = int(ratios[i+1]/ratios[i])
    ddmap = sum_pool(ddmap, ratio) 
    ddmaps.append(ddmap)
  return ddmaps
def sum_pool(arr, kernel_size=2):
  shape = arr.shape
  paddings = [[0, (kernel_size-shape[0]%kernel_size)%kernel_size], [0, (kernel_size-shape[1]%kernel_size)%kernel_size]]  
  if np.sum(paddings) > 0:
    arr = np.pad(arr, paddings, 'constant')
    shape = arr.shape
  new_shape = [int(shape[0]/kernel_size), kernel_size, int(shape[1]/kernel_size), kernel_size]
  return np.mean(arr.reshape(new_shape), axis=(1,3)) * (kernel_size * kernel_size)
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(round(img_width/input_width)))
  rows = max(1, int(round(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def get_resized_coords(coords, resize, img_size):
  resized_height, resized_width = resize
  img_height, img_width = img_size
  new_coords = []
  for coord in coords:
    y = int( min( coord[1]*resized_height/img_height, resized_height-1 ) )
    x = int( min( coord[0]*resized_width/img_width, resized_width-1 ) )
    new_coords.append([x,y])
  return new_coords
def get_coords_map(coords, height, width):
  coords_map = np.zeros([height, width])
  for coord in coords:
    coords_map[int(coord[1])][int(coord[0])] += 1
  return coords_map
def get_density_map(coords_map):
  return gaussian_filter(coords_map, sigma=10, mode='constant', cval=0.0, truncate=1.8)
def random_divide_crop(img, coords):
  input_height = 384
  input_width = 512
  width, height = img.size
  rows, columns = fit_grid(height, width, input_size=[input_height, input_width])
  resized_height = rows* input_height
  resized_width = columns* input_width
  crop_top = input_height*random.randint(0, rows-1)
  crop_left = input_width*random.randint(0, columns-1)
  crop_bottom = crop_top + input_height
  crop_right = crop_left + input_width
  coords = get_resized_coords(coords, (resized_height, resized_width), (img.size[1], img.size[0]))
  coords_map = get_coords_map(coords, resized_height, resized_width)
  top_pad = min(50, crop_top)
  left_pad = min(50, crop_left)
  right_pad = min(50, resized_width-crop_right)
  bottom_pad = min(50, resized_height-crop_bottom)
  density_map = get_density_map(coords_map[crop_top-top_pad:crop_bottom+bottom_pad, crop_left-left_pad:crop_right+right_pad])
  density_map = density_map[top_pad:top_pad+input_height, left_pad:left_pad+input_width]
  img_crop = img.resize((resized_width, resized_height)).crop((crop_left, crop_top, crop_right, crop_bottom))
  ddmaps = get_downsized_density_maps(density_map)
  return img_crop, ddmaps
def random_crop(img, coords):
  input_height = 384
  input_width = 512
  width, height = img.size
  crop_top = random.randint(0, max(0, height-input_height-1))
  crop_left = random.randint(0, max(0, width-input_width-1))
  crop_bottom = crop_top + input_height
  crop_right = crop_left + input_width
  coords_map = get_coords_map(coords, height, width)
  top_pad = min(50, crop_top)
  left_pad = min(50, crop_left)
  right_pad = min(50, width-crop_right)
  bottom_pad = min(50, height-crop_bottom)
  density_map = get_density_map(coords_map[crop_top-top_pad:crop_bottom+bottom_pad, crop_left-left_pad:crop_right+right_pad])
  density_map = density_map[top_pad:top_pad+input_height, left_pad:left_pad+input_width]
  img_crop = img.crop((crop_left, crop_top, crop_right, crop_bottom))
  ddmaps = get_downsized_density_maps(density_map)
  return img_crop, ddmaps 
def jitter_size(img, coords):
  input_height = 384
  input_width = 512
  width, height = img.size
  resized_width = max(int( width * ( 0.3 + random.random()*0.9 ) ), input_width)
  resized_height = max(int( height * resized_width/width * ( 0.7 + random.random()*0.6 ) ), input_height)
  coords = get_resized_coords(coords, (resized_height, resized_width), (height, width) )
  img = img.resize((resized_width, resized_height))
  return img, coords
def jitter_colour(image):
  random_factor = np.random.randint(5, 20) / 10. 
  color_image = ImageEnhance.Color(image).enhance(random_factor)
  random_factor = np.random.randint(5, 20) / 10.
  brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
  random_factor = np.random.randint(5, 20) / 10.
  contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
  random_factor = np.random.randint(5, 20) / 10.
  return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
def pad_image(img):
  input_height = 384
  input_width = 512
  width, height = img.size
  if width < input_width or height < input_height:
    img = ImageOps.expand(img, (0, 0, max(0, input_width-width), max(0, input_height-height) ))
  return img
def denormalise(img):
  img = np.transpose(img, (1,2,0))
  img *= [0.229, 0.224, 0.225]
  img += [0.485, 0.456, 0.406]
  img *= 255
  return img.astype('uint8')
def normalise(img, ddmaps=None):
  img = np.asarray(img, dtype='float32')
  img /= 255
  img -= [0.485, 0.456, 0.406]
  img /= [0.229, 0.224, 0.225]
  img = np.transpose(img, (2,0,1))
  if ddmaps is not None:
    ddmaps = [np.reshape(ddmap, (1,)+ddmap.shape).astype('float32') for ddmap in ddmaps]
    return img, ddmaps
  else:
    return img
def augment(img, coords):

  img = pad_image(img)
  
  factor = random.random()
  if factor < 0.9:
    img, coords = jitter_size(img, coords)

  factor = random.random()
  if factor < 0.3:
    img, ddmaps = random_divide_crop(img, coords)
  else:
    img, ddmaps = random_crop(img, coords)

  factor = random.random()
  if factor < 0.02:
    img = img.convert('L').convert('RGB')
  elif factor > 0.66:
    img = jitter_colour(img)

  factor = random.random()
  if factor < 0.5:
    img = ImageOps.mirror(img)
    ddmaps = [np.flip(ddmap, axis=1) for ddmap in ddmaps]

  img, ddmaps = normalise(img, ddmaps)
  return img, ddmaps
def next_batch(batch_size, names):
  ids = np.random.randint(0, len(names)-1, [batch_size])
  names = names[ids]

  imgs = []
  targets15 = []
  targets14 = []
  targets13 = []
  targets12 = []
  targets11 = []
  targets10 = []

  threadLock = Lock()

  def threadjob(name):
    img = Image.open(name+'.jpg')
    img.load()
    coords = pickle.load(open(name+'.pkl'))
    #print('coord: ', coords, name)
    img, ddmaps = augment(img, coords)
    threadLock.acquire()
    imgs.append(img)
    target10, target11, target12, target13, target14 = ddmaps
    targets15.append(np.reshape(np.sum(target14), [1,1,1]))
    targets14.append(target14)
    targets13.append(target13)
    targets12.append(target12)
    targets11.append(target11)
    targets10.append(target10)
    threadLock.release()
  threads = []
  for name in names:
    thread = Thread(target=threadjob, args=(name,))
    thread.start()
    threads.append(thread)
  for thread in threads:
    thread.join()

  targets = [targets15, targets14, targets13, targets12, targets11, targets10]

  targets = [np.array(tg, dtype='float32') for tg in targets]
  imgs = np.array(imgs, dtype='float32')
  return imgs, targets
