import os
import zipfile
import urllib.request

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import time
import matplotlib.pyplot as plt

#def get_data(path):
#    if not os.path.exists("/content/drive/MyDrive/Colab_Notebooks/Observation-earth/IPEO_Planet_project.zip"):
#            print(f"downloading IPEO_Planet_project.zip")
#            urllib.request.urlretrieve("https://drive.google.com/file/d/1zS6wtrRfTeLaHwlS0HDlNiAc_b_ZrTeX/view?usp=share_link",
#                                       "/content/drive/MyDrive/Colab_Notebooks/Observation-earth/IPEO_Planet_project.zip")
#    else:
#        print(3)
#    with zipfile.ZipFile(path, 'r') as zip_ref:
#        zip_ref.extractall()


class DatasetAmazon(Dataset):
    def __init__(self, full=False, tiny=False, test=False, val=False, split=[0.6, 0.15, 0.25], path_to_labels=None):
        """images dataset initialization

        Args:
            full (bool, optional): Use the full set. Defaults to False.
            tiny (bool, optional): Use a very small set for quick checks.
            Defaults to False.
            test (bool, optional): load test data. Defaults to False.
            val (bool, optional): load validation data. Defaults to False.
            split (list[floats], optional): how to split the data (the sum needs to be equal to one)
        """
        self.full = full
        self.tiny = tiny
        self.test = test
        self.val = val
        self.split = split
        if not sum(self.split) == 1.0:
            raise ValueError('The splitting procedure needs to equal 1')
        if not len(self.split) == 3:
            raise ValueError('You need to provide a splitting value for the train, val, and test (3 values)')

        ###### A FAIRE PLUS INTELLIGEMMENT #####
        train_stop = int(np.floor(self.split[0]*40479))
        val_stop = int(np.floor(self.split[0]+self.split[1]*40479))
        
        self.SPLITS = {
        'train': list(range(0, train_stop)),    
        'val':   list(range(train_stop+1, val_stop)),   
        'test':  list(range(val_stop+1, 40479))   
        }
        ########################################
        self.LABEL_CLASSES = pd.read_pickle(path_to_labels)
        self.load_data()

    def __getitem__(self, index):
        #t1 = time.time()
        imgName, label = self.data[index]
        im = plt.imread(imgName)
        print(im.shape)
        im = im.transpose(2, 0, 1)
        print(im.shape)
        img = torch.from_numpy(np.array(im))
        #print("getitem: ", time.time()-t1)
        return img.float(), label

    def __len__(self):
        return len(self.data)

    def load_data(self):
        t = time.time()
        data_loc = "../IPEO_Planet_project/"
        self.data = []                                  # list of tuples of (image path, label class)
        #transform = T.ToTensor()
        if self.val:
            for imgIndex in self.SPLITS['val']:
                imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                # example format: 'baseFolder/agricultural/agricultural07.tif'
                #img = plt.imread(imgName)
                #img = transform(img)
                self.data.append((
                    imgName,
                    self.LABEL_CLASSES.iloc[imgIndex].values         # get index for label class
                ))
        elif self.test:
            for imgIndex in self.SPLITS['test']:
                imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                # example format: 'baseFolder/agricultural/agricultural07.tif'
                #img = plt.imread(imgName) #img = Image.open(imgName)
                #img = transform(img)
                self.data.append((
                    imgName,
                    self.LABEL_CLASSES.iloc[imgIndex].values         # get index for label class
                ))
        else:
            for imgIndex in self.SPLITS['train']:
                imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                # example format: 'baseFolder/agricultural/agricultural07.tif'
                #img = plt.imread(imgName) #img = Image.open(imgName)
                #img = transform(img)
                self.data.append((
                    imgName,
                    self.LABEL_CLASSES.iloc[imgIndex].values         # get index for label class
                ))
        if self.full:
            if self.tiny:
                raise ValueError('Cannot have both --full and --tiny')
        else:
            if self.tiny:
                print('** Reduce the data-set to the tiny setup (factor of 1000)')
                self.data = self.data[0:1000:-1]
            else:
                print('** Reduce the data-set by a factor 100 (use --full for the full thing)')
                self.data = self.data[0:100:-1]
        print("loading took ", time.time()-t)



