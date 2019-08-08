# deepcount

Deep Density-aware Count Regressor
- a state-of-the-art method for crowd counting

### Installation
1. Install environment: Python 2.7, Tensorflow >= 1.8, PyTorch >= 0.4, Pillow
2. Download pretrained vgg model from https://download.pytorch.org/models/vgg16-397923af.pth
3. Place vgg16-397923af.pth under the work directory

### Data
Prepare the dataset first and modify data.py to get the data accessible. See data.py for detail.
The current code of data.py is for Shanghai Tech Part B. Before training the model download 

### Run
```
python train.py
```
