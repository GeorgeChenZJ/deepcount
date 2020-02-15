# deepcount

Deep Density-aware Count Regressor
- a state-of-the-art method for crowd counting
- fast, easy to implement and scale
- paper: https://arxiv.org/abs/1908.03314

### PaddlePaddle Implementation Branch
https://github.com/GeorgeChenZJ/deepcount/tree/paddle

### Installation
1. Install requirements: Python 2.7, Tensorflow >= 1.8, PyTorch >= 0.4, Pillow, tqdm, scipy
2. Download pretrained vgg model from https://download.pytorch.org/models/vgg16-397923af.pth
3. Place vgg16-397923af.pth under the working directory

### Data
- Prepare the dataset first and modify data.py to make the data accessible. See data.py for detail.
The current code of data.py is for Shanghai Tech Part B. Before training the model download Shanghai Tech dataset should be downloaded (https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view). Change 'data_dir' in data.py and relevant codes for loading the dataset.

### Run
- Set batch size and number of devices in train.py and then run
```
python train.py
```
- Note that the code is written to be plain but not optimally efficient.

### Trained Weights
- We provide trained model on ShanghaiTech Part B: https://drive.google.com/open?id=1qaYXLX5vYS0prhDFdYLWSr4YFDf_geQz
