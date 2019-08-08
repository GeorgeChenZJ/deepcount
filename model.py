from __future__ import print_function, division
import tensorflow as tf
import numpy as np

def conv(kernel_size, input, filters, padding='same', strides=(1,1), name=None, act=tf.nn.relu, dilation=1, dropout=None, training=True):
  out = tf.layers.conv2d(input, filters, kernel_size,
                        strides=strides,
                        dilation_rate = 1,
                        activation=act,
                        padding=padding,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                        name=name)
  if dropout is not None:
    out = tf.layers.dropout(out, dropout, training=training)
  return out
def conv_t(kernel_size, input, filters, strides=2, padding='same', act=tf.nn.leaky_relu, dropout=None, training=True):
  out = tf.layers.conv2d_transpose(input, filters, kernel_size, padding=padding, strides=strides, activation=act,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),)
  if dropout is not None:
    out = tf.layers.dropout(out, dropout, training=training)
  return out
def maxpool(kernel_size, input, strides=2):
  return tf.layers.max_pooling2d(input, kernel_size, strides, padding='same')
def abs_loss(predict, target):
  loss = tf.losses.absolute_difference(target, predict, reduction=tf.losses.Reduction.NONE)
  return tf.reduce_mean(loss)
def pdims(tensor):
  return np.prod(tensor.get_shape().as_list()[1:])
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
def encoder(input, training, dropout=0):
  # input: 384x512
  layer1 = conv(3, input, 64, name='vgg_conv_1')
  layer2 = conv(3, layer1, 64, name='vgg_conv_2')
  pool = maxpool(2, layer2)
  layer3 = conv(3, pool, 128, name='vgg_conv_3')
  layer4 = conv(3, layer3, 128, name='vgg_conv_4')
  pool = maxpool(2, layer4)
  layer5 = conv(3, pool, 256, name='vgg_conv_5')
  layer6 = conv(3, layer5, 256, name='vgg_conv_6')
  layer7 = conv(3, layer6, 256, name='vgg_conv_7') # 96x128 4
  pool = maxpool(2, layer7)
  layer8 = conv(3, pool, 512, name='vgg_conv_8')
  layer9 = conv(3, layer8, 512, name='vgg_conv_9')
  layer10 = conv(3, layer9, 512, name='vgg_conv_10') # 48x64 8

  layer10 = conv(3, layer10, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu) # 48x64 8
  print('10', layer10.shape) # 24x32 16

  layer11 = conv(3, layer10, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer11 = conv(3, layer11, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('11', layer11.shape) # 24x32 16
  layer12 = conv(3, layer11, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer12 = conv(3, layer12, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('12', layer12.shape) # 12x16 32
  layer13 = conv(3, layer12, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer13 = conv(3, layer13, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('13', layer13.shape) # 6x8  64
  layer14 = conv(3, layer13, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer14 = conv(3, layer14, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('14', layer14.shape) # 3x4  128
  layer15 = conv((3,4), layer14, 1024, padding='valid', dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('15', layer15.shape) # 1  a

  return layer10, layer11, layer12, layer13, layer14, layer15
def decoder(inputs, training, dropout):
  layer10, layer11, layer12, layer13, layer14, layer15 = inputs

  out15 = conv(1, layer15, 1, act=tf.nn.leaky_relu)
  print('out15', out15.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  out14 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out14', out14.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  out13 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out13', out13.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  out12 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out12', out12.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer11], axis=3)
  out11 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out11', out11.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer11], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer10], axis=3)
  out10 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out10', out10.shape)

  return out15, out14, out13, out12, out11, out10

def en_decode(input, training, dropout, reuse):
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    Encoded = encoder(input, training, dropout)
    Decoded = decoder(Encoded, training, dropout)
    return Decoded

def model(input, targets, training, alpha, dropout=0.3, gpu_num=0):

  target15, target14, target13, target12, target11, target10 = targets

  print('input:', input.shape)

  input = tf.split(input, gpu_num)
  target15 = tf.split(target15, gpu_num)
  target14 = tf.split(target14, gpu_num)
  target13 = tf.split(target13, gpu_num)
  target12 = tf.split(target12, gpu_num)
  target11 = tf.split(target11, gpu_num)
  target10 = tf.split(target10, gpu_num)

  medium = []

  optimizer_vgg = tf.train.AdamOptimizer(tf.maximum(alpha/2, 1e-7))
  optimizer = tf.train.AdamOptimizer(alpha)

  for gpu_id in range(int(gpu_num)):
      reuse = gpu_id > 0
      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
          with tf.name_scope('tower_%d' % gpu_id):

            Decoded = en_decode(input[gpu_id], training, dropout, reuse)
            out15, out14, out13, out12, out11, out10 = Decoded

            loss = 0
            loss += abs_loss(out15, target15[gpu_id]) * pdims(out15) * 16
            loss += abs_loss(out14, target14[gpu_id]) * pdims(out14) * 1
            loss += abs_loss(out13, target13[gpu_id]) * pdims(out13)
            loss += abs_loss(out12, target12[gpu_id]) * pdims(out12)
            loss += abs_loss(out11, target11[gpu_id]) * pdims(out11)
            loss += abs_loss(out10, target10[gpu_id]) * pdims(out10)
            loss /= 100

            trainables = tf.trainable_variables()
            grads_vgg = optimizer_vgg.compute_gradients(loss, var_list=[var for var in trainables if 'vgg' in var.name])
            grads = optimizer.compute_gradients(loss, var_list=[var for var in trainables if 'vgg' not in var.name])

            medium.append((loss, Decoded, grads_vgg, grads))

  L2_loss = tf.losses.get_regularization_loss() * 1e-4

  losses, Decoded_all, grads_vgg, grads = zip(*medium)

  loss = tf.reduce_mean(losses)
  loss += L2_loss

  train_vgg = optimizer_vgg.apply_gradients(average_gradients(grads_vgg))
  train_others = optimizer.apply_gradients(average_gradients(grads))
  train = tf.group(train_vgg, train_others)

  D = []
  for i in range(len(Decoded_all[0])):
      outs = [Decoded_all[j][i] for j in range(len(Decoded_all))]
      outs = tf.concat(outs, axis=0)
      D.append(tf.nn.relu(outs))

  m = L2_loss
  return train, loss, D, m
