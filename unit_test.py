from data import *
import time
from PIL import Image

train_names, _ = get_train_data_names()

def display_set_of_imgs(images, rows=2, size=0.5, name='0'):
  n_images = len(images)
  with open('./visualise_images/'+str(name)+'.pkl', 'wb') as f:
    pickle.dump(images, f)

for i in range(1):

  start = time.time()
  imgs, targets = next_batch(20, train_names)
  #Image.fromarray(denormalise(imgs[0])).save('./visualise_images/img.jpg')
  #display_set_of_imgs([target[0] for target in  targets] )
  print(time.time()-start)
