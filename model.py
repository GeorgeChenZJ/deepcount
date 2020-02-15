from __future__ import print_function, division
import paddle
import paddle.fluid.layers as pd
import numpy as np
import os
from threading import Thread
import pickle

def conv(kernel_size, input, filters, padding=1, strides=(1,1), name=None, act='relu', dilation=1, dropout=None, training=False):
  if kernel_size==1:
    padding = 0
  if act=='leaky_relu':
    activation = None
  else:
    activation = act
  out = pd.conv2d(input, filters, kernel_size,
                        stride = strides,
                        dilation = dilation,
                        act=activation,
                        padding=padding,
                        param_attr=paddle.fluid.initializer.Xavier(),
                        bias_attr=paddle.fluid.initializer.Constant(0.0), name=name)
  if act=='leaky_relu':
    out = pd.leaky_relu(out, alpha=0.2)
  return out
def conv_t(kernel_size, input, filters, output=None, strides=2, padding=1, act='leaky_relu', dropout=None, training=False, name=None):
  out = input
  if act=='leaky_relu':
    activation = None
  else:
    activation = act
  if output:
    padding = 0
    stride = 1
  out = pd.conv2d_transpose(out, filters, filter_size=kernel_size,                       
                        padding=padding, stride=strides, 
                        output_size=output,                        
                        act=activation,
                        param_attr=paddle.fluid.initializer.Xavier(),
                        bias_attr=paddle.fluid.initializer.Constant(0.0), name=name)
  if act=='leaky_relu':
    out = pd.leaky_relu(out, alpha=0.2)
  return out
def maxpool(kernel_size, input, strides=2):
  return pd.pool2d(input, pool_size=kernel_size, pool_type='max', pool_stride=strides)
def abs_loss(predict, target):
  loss = pd.abs(target-predict)
#  loss = pd.reduce_sum(loss, [1,2,3])
  loss = pd.reduce_mean(loss)
  return loss * target.shape[2] * target.shape[3]
def encoder(input, training, dropout=0):
  # input: 384x512
  layer = conv(3, input, 64, name='vgg_conv_1')
  layer = conv(3, layer, 64, name='vgg_conv_2')
  pool = maxpool(2, layer)
  layer = conv(3, pool, 128, name='vgg_conv_3')
  layer = conv(3, layer, 128, name='vgg_conv_4')
  pool = maxpool(2, layer)
  layer = conv(3, pool, 256, name='vgg_conv_5')
  layer = conv(3, layer, 256, name='vgg_conv_6')
  layer = conv(3, layer, 256, name='vgg_conv_7') # 96x128 4
  pool = maxpool(2, layer)
  layer = conv(3, pool, 512, name='vgg_conv_8')
  layer = conv(3, layer, 512, name='vgg_conv_9')
  layer10 = conv(3, layer, 512, name='vgg_conv_10') # 48x64 8

  layer10 = conv(3, layer10, 256, strides=1, dropout=dropout, training=training, act='leaky_relu') # 48x64 8
  print('10', layer10.shape) # 24x32 16

  layer11 = conv(3, layer10, 512, strides=2, dropout=dropout, training=training, act='leaky_relu')
  layer11 = conv(3, layer11, 256, strides=1, dropout=dropout, training=training, act='leaky_relu')
  print('11', layer11.shape) # 24x32 16
  layer12 = conv(3, layer11, 512, strides=2, dropout=dropout, training=training, act='leaky_relu')
  layer12 = conv(3, layer12, 256, strides=1, dropout=dropout, training=training, act='leaky_relu')
  print('12', layer12.shape) # 12x16 32
  layer13 = conv(3, layer12, 512, strides=2, dropout=dropout, training=training, act='leaky_relu')
  layer13 = conv(3, layer13, 256, strides=1, dropout=dropout, training=training, act='leaky_relu')
  print('13', layer13.shape) # 6x8  64
  layer14 = conv(3, layer13, 512, strides=2, dropout=dropout, training=training, act='leaky_relu')
  layer14 = conv(3, layer14, 256, strides=1, dropout=dropout, training=training, act='leaky_relu')
  print('14', layer14.shape) # 3x4  128
  layer15 = conv((3,4), layer14, 1024, padding=0, dropout=dropout, training=training, act='leaky_relu')
  print('15', layer15.shape) # 1  a

  return layer10, layer11, layer12, layer13, layer14, layer15
def decoder(inputs, training, dropout):
  layer10, layer11, layer12, layer13, layer14, layer15 = inputs

  out15 = conv(1, layer15, 1, act='relu')
  print('out15', out15.shape)

  layer = conv_t((3,4), layer15, 256, output=[3,4], padding='valid', strides=1, dropout=dropout, training=training)
  layer = pd.concat([layer, layer14], axis=1)
  out14 = conv(1, layer, 1, act='relu')
  print('out14', out14.shape)

  layer = conv_t((3,4), layer15, 256, output=[3,4], padding='valid', strides=1, dropout=dropout, training=training)
  layer = pd.concat([layer, layer14], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer13], axis=1)
  out13 = conv(1, layer, 1, act='relu')
  print('out13', out13.shape)

  layer = conv_t((3,4), layer15, 256, output=[3,4], padding='valid', strides=1, dropout=dropout, training=training)
  layer = pd.concat([layer, layer14], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer13], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer12], axis=1)
  out12 = conv(1, layer, 1, act='relu')
  print('out12', out12.shape)

  layer = conv_t((3,4), layer15, 256, output=[3,4], padding='valid', strides=1, dropout=dropout, training=training)
  layer = pd.concat([layer, layer14], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer13], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer12], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer11], axis=1)
  out11 = conv(1, layer, 1, act='relu')
  print('out11', out11.shape)

  layer = conv_t((3,4), layer15, 256, output=[3,4], padding='valid', strides=1, dropout=dropout, training=training)
  layer = pd.concat([layer, layer14], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer13], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer12], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer11], axis=1)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = pd.concat([layer, layer10], axis=1)
  out10 = conv(1, layer, 1, act='relu')
  print('out10', out10.shape)

  return out15, out14, out13, out12, out11, out10

def model(input, targets, alpha, training=False, dropout=None):

  target15, target14, target13, target12, target11, target10 = targets

  print('input:', input.shape)

  Encoded = encoder(input, training, dropout)
  Decoded = decoder(Encoded, training, dropout)

  out15, out14, out13, out12, out11, out10 = Decoded

  loss = 0
  loss += abs_loss(out15, target15) * 16
  loss += abs_loss(out14, target14)
  loss += abs_loss(out13, target13)
  loss += abs_loss(out12, target12)
  loss += abs_loss(out11, target11)
  loss += abs_loss(out10, target10)
  loss /= 100

  m = out15

  test_program = paddle.fluid.default_main_program().clone(for_test=True)

  optimizer = paddle.fluid.optimizer.AdamOptimizer(5e-6,
                        regularization=paddle.fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-5))
  optimizer.minimize(loss)

  return loss, out15, out14, out13, out12, out11, out10, test_program, m

class Saver:
  def __init__(self, exe, path='./', max_to_keep=1):
    if not path.endswith('/'):
      path += '/'
    if not os.path.exists(path):
      os.makedirs(path)
    self.exe = exe
    self.path = path
    self.max_to_keep = max_to_keep
    self.checkpoint_path = path+'checkpoint.pkl'
    if not os.path.exists(self.checkpoint_path):
      checkpoints = []
    else:
      checkpoints = self._read_checkpoints()
      if len(checkpoints) > max_to_keep:
        checkpoints = checkpoints[-max_to_keep:]
    self._write_checkpoints(checkpoints)
  def _write_checkpoints(self, checkpoints):
    def _write(checkpoints):
      with open(self.checkpoint_path, 'wb') as f:
        pickle.dump(checkpoints, f)
    thread = Thread(target=_write, args=(checkpoints,))
    thread.start()
    thread.join()
  def _read_checkpoints(self):
    try:
      with open(self.checkpoint_path, 'rb') as f:
        checkpoints = pickle.load(f)
    except EOFError:
      checkpoints = []
    return checkpoints
  def add_checkpoint(self, name):
    checkpoints = self._read_checkpoints()
    if len(checkpoints)==self.max_to_keep:
      name_to_delete = checkpoints.pop(0)
      self._delete_checkpoint(name_to_delete, checkpoints)
    checkpoints.append(name)
    self._write_checkpoints(checkpoints)
  def _delete_checkpoint(self, name_to_delete, checkpoints):
    if not name_to_delete in checkpoints and os.path.exists(self.path+name_to_delete):
      os.remove(self.path+name_to_delete)
  def last_checkpoint(self, n=-1):
    checkpoints = self._read_checkpoints()
    assert (n<0 and -n<=len(checkpoints)) or (n>=0 and n<len(checkpoints)-1), "Invalid checkpoint index: "+str(n)
    return checkpoints[n]
  def save(self, name):
    paddle.fluid.io.save_persistables(executor=self.exe, dirname=self.path, filename=name )
    self.add_checkpoint(name)
  def restore(self, name):
    print('INFO: Restoring parameters from', self.path+name)
    paddle.fluid.io.load_persistables(executor=self.exe, dirname=self.path, filename=name )
