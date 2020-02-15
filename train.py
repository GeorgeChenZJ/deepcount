from __future__ import print_function, division
import paddle
import paddle.fluid.layers as pd
import time
import logging
import random
from threading import Thread

from functions import *
from model import *
from test import *
from data import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('resume', nargs='?', default='0')
args = parser.parse_args()

import os, sys
#os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_num = 1
new_model = True
#batch_size = int(sys.argv[1])
batch_size = 4

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

saver = Saver(exe=exe, path='./output/model', max_to_keep=2)
best_saver = Saver(exe=exe, path='./output/best_model', max_to_keep=1)
#   print('total number of parameters:', total_parameters())

logging.basicConfig(filename='./output/train.log',level=logging.INFO)
train_names, test_names = get_train_data_names()

if True:
  print("Training begins")
  if new_model:
    exe.run(program=paddle.fluid.default_startup_program())
    set_pretrained(place)
    global_step = 0
    EMA = 0
    train_MAEs = None
    test_MAEs = None
    best_result = 200
  else:
    last_cp = saver.last_checkpoint()
    saver.restore( last_cp )
    global_step = int(last_cp[last_cp.rfind('-')+1:])
#    global_step = 400000
    EMA = 0
    train_MAEs = None
    test_MAEs = None
#    best_result = best_saver.last_checkpoint()
#    best_result = float(best_result[best_result.rfind('-')+1:])
    best_result = 0.195
  try:

    batch_data = {}
    def data_loader(batch_names):
        batch_inputs, batch_targets = next_batch(batch_size, batch_names)
        batch_data['inputs'] = batch_inputs
        batch_data['targets'] = batch_targets
    train_data_thread = Thread(target=data_loader, args=(train_names,))
    train_data_thread.start()

    for step in range(global_step, 200000):
      if step < 20000:
        lr = 5e-5
      elif step < 40000:
        lr = 2e-5
      elif step < 60000:
        lr = 1e-5
      elif step < 80000:
        lr = 7e-6
      elif step < 100000:
        lr = 3e-6
      elif step < 120000:
        lr = 1e-6
      elif step < 160000:
        lr = 5e-7
      else:
        lr = 1e-7
#      if step%50000==0 and not step==0:
#        best_saver.restore( best_saver.last_checkpoint() )

      train_data_thread.join()
      prefetched_batch_data = batch_data
      train_inputs = prefetched_batch_data['inputs']
      train_targets = prefetched_batch_data['targets']

      batch_data = {}
      train_data_thread = Thread(target=data_loader, args=(train_names,))
      train_data_thread.start()

      train_t15, train_t14, train_t13, train_t12, train_t11, train_t10 = train_targets

      [[train_loss]] = exe.run(fetch_list=[loss],
        program=main_program,
        feed={
          'input': train_inputs,
          'target15': train_t15,
          'target14': train_t14,
          'target13': train_t13,
          'target12': train_t12,
          'target11': train_t11,
          'target10': train_t10,
          'alpha': lr
      })
      if EMA == 0:
        EMA = train_loss
      else:
        EMA = moving_average(train_loss, EMA)
      if step%50==0:
        [train_loss], train_out15, train_out14, train_out13, train_out12, train_out11, train_out10, train_m = exe.run(
          program=test_program,
          fetch_list=[loss, out15, out14, out13, out12, out11, out10, monitor],
          feed={
            'input': train_inputs,
            'target15': train_t15,
            'target14': train_t14,
            'target13': train_t13,
            'target12': train_t12,
            'target11': train_t11,
            'target10': train_t10,
            'alpha': lr
        })
        train_D = train_out15, train_out14, train_out13, train_out12, train_out11, train_out10
        train_D = [np.maximum(dd, np.zeros_like(dd)) for dd in  train_D]

        test_inputs, test_targets = next_batch(batch_size, test_names)
        test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
        [test_loss], test_out15, test_out14, test_out13, test_out12, test_out11, test_out10, test_m = exe.run(
          program=test_program,
          fetch_list=[loss, out15, out14, out13, out12, out11, out10, monitor],
          feed={
            'input': test_inputs,
            'target15': test_t15,
            'target14': test_t14,
            'target13': test_t13,
            'target12': test_t12,
            'target11': test_t11,
            'target10': test_t10,
            'alpha': lr
        })
        test_D = test_out15, test_out14, test_out13, test_out12, test_out11, test_out10
        test_D = [np.maximum(dd, np.zeros_like(dd)) for dd in  test_D]

        print('>>>______', np.maximum(test_m[0], np.zeros_like(test_m[0])), test_t15[0])
        if train_MAEs is None:
          train_MAEs = [ MAE(train_D[c], train_targets[c]) for c in range(len(train_D)) ]
        else:
          train_MAEs = moving_average_array( [ MAE(train_D[c], train_targets[c]) for c in range(len(train_D)) ] , train_MAEs)
        if test_MAEs is None:
          test_MAEs = [ MAE(test_D[c], test_targets[c]) for c in range(len(test_D)) ]
        else:
          test_MAEs = moving_average_array( [ MAE(test_D[c], test_targets[c]) for c in range(len(test_D)) ] , test_MAEs)

        log_str = ['>>> TRAIN', time.asctime()[10:20]+': i [', str(global_step), '] || [loss, EMA]: [',
                   str(round(train_loss, 2))+', ', str(round(EMA,2)), '] || [EMAoMAE]:', str(train_MAEs)]
        print(*log_str)
        logging.info(' '.join(log_str))

        log_str = ['>>> TEST ', time.asctime()[10:20]+': i [', str(global_step), '] || [EMAoMAE]:', str(test_MAEs)]
        print(*log_str)
        logging.info(' '.join(log_str))

        if step%400==0 and False:

          display_set_of_imgs([train_D[1][0], train_targets[1][0], train_D[2][0], train_targets[2][0], train_D[3][0]
                               , train_targets[3][0], train_D[4][0], train_targets[4][0], train_out10[5][0], train_t10[5][0]
                               , denormalize(train_inputs[0])], rows=3, size=2)
          display_set_of_imgs([test_D[1][0], test_targets[1][0], test_D[2][0], test_targets[2][0], test_D[3][0]
                               , test_targets[3][0], test_D[4][0], test_targets[4][0], test_out10[5][0], test_t10[5][0]
                               , denormalize(test_inputs[0])], rows=3, size=2)

        if step%2000==0:
          saver.save("model-"+str(global_step))
          log_str = [">>> Model saved:", str(global_step)]
          print(*log_str)
          logging.info(' '.join(log_str))
          if True:#global_step>=2000 or step==0:
            test_results = full_test(exe, test_program,
                fetch_list=[out15, out14, out13, out12, out11, out10],
                gpu_num=gpu_num)
            log_str = ['>>> TEST ', time.asctime()+': i [', str(global_step),
                       '] || [Result]:', str([round(result, 4) for result in test_results])]
            if round(test_results[0],4) < best_result:
              best_result = round(test_results[0],4)
              best_saver.save("model-"+str(best_result))
              log_str.append(' * BEST *')
            print(*log_str)
            logging.info(' '.join(log_str))

      global_step = global_step + 1
  except KeyboardInterrupt:
    print('>>> KeyboardInterrupt. Saving model...')
    saver.save("model-"+str(global_step))
    print(">>> Model saved:", str(global_step))

