import os
import pickle
pickle.HIGHEST_PROTOCOL = 3

import pandas as pd
#from torchvision.io import read_image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
# from skimage import io
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import torch
from utils.utility import *
import cv2 
import ast 
from torchvision import transforms
from scipy.special import expit, logit
#from tree_loss import * 

seed=999
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ImageDataset(Dataset):
    def __init__(self, params,mode):

        self.input_size=int(params['input_size'])
        self.num_classes=int(params['num_classes'])
        self.mode=mode

        self.pre_process_dataset(params)
        

    def pre_process_dataset(self,params):

        labels_files=ast.literal_eval(params[self.mode+'_labels_files'])
        images_dirs=ast.literal_eval(params[self.mode+'_images_folders'])
        
        #self.all_images=images_dirs[0]+self.img_labels.iloc[:,-1].to_numpy()
        self.all_img_ids=np.empty(shape=(0,))
        self.all_labels=np.empty(shape=(0,self.num_classes),dtype=np.float32)
        self.all_images=np.empty(shape=(0,))

        for i in range(len(labels_files)):
            img_labels=pd.read_hdf(labels_files[i],key='df',mode='r')
            self.all_img_ids=np.concatenate((self.all_img_ids,img_labels.iloc[:,-1].to_numpy()),axis=0)
            self.all_labels=np.concatenate((self.all_labels,img_labels.iloc[:,:-1].to_numpy()),axis=0)
            self.all_images=np.concatenate((self.all_images,images_dirs[i]+img_labels.iloc[:,-1].to_numpy()),axis=0)

        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(self.input_size),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                #transforms.Normalize(meanstds['means'], meanstds['stds'])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                #transforms.Normalize(meanstds['means'], meanstds['stds'])
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                #transforms.Normalize(meanstds['means'], meanstds['stds'])
                #transforms.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])
            ])
        }
        self.transform = data_transforms[self.mode]

        self.all_labels=np.where(self.all_labels>0,1,0)
        print(self.all_labels.shape,self.all_images.shape,self.all_img_ids.shape)


    def __len__(self):
        return len(self.all_images)
        #return len(self.selected_images)

    def __getitem__(self, idx):
        img_path=self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = np.array(self.all_labels[idx,:],dtype=np.float32)

        return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)

        #return self.get_function(idx)

