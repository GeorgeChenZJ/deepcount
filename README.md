# deepcount

Deep Density-aware Count Regressor
- a state-of-the-art method for crowd counting

### Installation
1. Install requirements: Python 2.7, Tensorflow >= 1.8, PyTorch >= 0.4, Pillow, tqdm, scipy
2. Download pretrained vgg model from https://download.pytorch.org/models/vgg16-397923af.pth
3. Place vgg16-397923af.pth under the work directory

### Data
- Prepare the dataset first and modify data.py to make the data accessible. See data.py for detail.
The current code of data.py is for Shanghai Tech Part B. Before training the model download Shanghai Tech dataset should be downloaded (https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view). Change 'data_dir' in data.py to correct path for the dataset.

### Run
- Set batch size and number of devices in train.py and then run
```
python train.py
```
- The code is written to be plain but not optimally efficient
