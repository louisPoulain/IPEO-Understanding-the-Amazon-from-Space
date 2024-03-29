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


def get_all_indexes():
    index_file = np.load("List_label_rare/list_label_rare.npy", allow_pickle=True)
    return index_file


class DatasetAmazon(Dataset):
    def __init__(self, full=False, tiny=False, test=False, val=False, split=[0.6, 0.15, 0.25], path_to_labels=None, read_npy=False):
        """images dataset initialization

        Args:
            full (bool, optional): Use the full set. Defaults to False.
            tiny (bool, optional): Use a very small set for quick checks.
            Defaults to False.
            test (bool, optional): load test data. Defaults to False.
            val (bool, optional): load validation data. Defaults to False.
            split (list[floats], optional): how to split the data (the sum needs to be equal to one)
            read_npy = whether images should be taken from the npy file or not
        """
        self.full = full
        self.tiny = tiny
        self.test = test
        self.val = val
        self.split = split
        self.read_npy = read_npy
        if not sum(self.split) == 1.0:
            raise ValueError('The splitting procedure needs to equal 1')
        if not len(self.split) == 3:
            raise ValueError('You need to provide a splitting value for the train, val, and test (3 values)')

        # Separate rare and frequent indexes
        all_ind = get_all_indexes()
        labels = ["blow_down", "conventional_mine", "slash_burn", "blooming", 
                    "artisinal_mine", "select_logging", "bare_ground", "frequent"]
        dict_labels = dict([(labels[i], all_ind[i]) for i in range(8)])          

        # Splitting procedure
        for label in dict_labels.keys():
            tmp = dict_labels[label]
            np.random.shuffle(tmp)
            l = len(tmp)
            train_stop = int(np.floor(self.split[0]*l))
            val_stop = int(np.floor((self.split[0]+self.split[1])*l))
            SPLITS = {
                'train': tmp[:train_stop],    
                'val':   tmp[train_stop+1:val_stop],   
                'test':  tmp[val_stop+1:l]
                }
            dict_labels[label] = SPLITS
        
        self.SPLITS = dict_labels
        self.all_ind = all_ind
        ########################################
        self.LABEL_CLASSES = pd.read_pickle(path_to_labels)
        self.load_data()

    def __getitem__(self, index):
        #t1 = time.time()
        if self.read_npy:
            imgName, imgIndex, label = self.data[index]
            im = np.load(imgName)[:, :, :, imgIndex]
        else:
            imgName, label = self.data[index]
            im = plt.imread(imgName)
        im = im.transpose(2, 0, 1)
        img = torch.from_numpy(np.array(im))
        #print("getitem: ", time.time()-t1)
        return img.float(), label

    def __len__(self):
        return len(self.data)

    def load_data(self):
        t = time.time()
        data_loc = "../IPEO_Planet_project/"
        self.data = []                                  # list of tuples of (image path, label class)
        #if self.val:
        #    nb_img = sum(map(len, self.all_ind))
        #    print(f"Loading validation data ({self.SPLITS['val'].shape[0]} images)")
        #print(f"Loading test data ({self.SPLITS['test'].shape[0]} images)")
        #print(f"Loading training data ({self.SPLITS['train'].shape[0]} images)")
        for label in self.SPLITS.keys():
            if self.val:
                for imgIndex in self.SPLITS[label]['val']:
                    if self.read_npy:
                        imgName = os.path.join(data_loc,
                                f'train-jpg/numpy_ndarray_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}.npy')
                        self.data.append((imgName, imgIndex-(imgIndex//1000)*1000, self.LABEL_CLASSES.iloc[imgIndex].values))
                    else:
                        imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                        self.data.append((imgName, self.LABEL_CLASSES.iloc[imgIndex].values))
            elif self.test:
                for imgIndex in self.SPLITS[label]['test']:
                    if self.read_npy:
                        imgName = os.path.join(data_loc,
                                f'train-jpg/numpy_ndarray_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}.npy')
                        self.data.append((imgName, imgIndex-(imgIndex//1000)*1000, self.LABEL_CLASSES.iloc[imgIndex].values))
                    else:
                        imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                        self.data.append((imgName, self.LABEL_CLASSES.iloc[imgIndex].values))
            else:
                for imgIndex in self.SPLITS[label]['train']:
                    if self.read_npy:
                        imgName = os.path.join(data_loc,
                                f'train-jpg/numpy_ndarray_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}.npy')
                        self.data.append((imgName, imgIndex-(imgIndex//1000)*1000, self.LABEL_CLASSES.iloc[imgIndex].values))
                    else:
                        imgName = os.path.join(data_loc, 
                                f'train-jpg/train_{(imgIndex//1000)*1000}-{(imgIndex//1000+1)*1000-1}/train_{str(imgIndex)}.jpg') 
                        self.data.append((imgName, self.LABEL_CLASSES.iloc[imgIndex].values))
        if self.full:
            if self.tiny:
                raise ValueError('Cannot have both --full and --tiny')
        else:
            if self.tiny:
                print('** Reduce the data-set to the tiny setup (factor of 1000)')
                self.data = self.data[0:-1:1000]
            else:
                print('** Reduce the data-set by a factor 100 (use --full for the full thing)')
                self.data = self.data[0:-1:100]
        print("loading took ", time.time()-t)



