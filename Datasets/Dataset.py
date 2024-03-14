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
torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)




# class ImageDataset(Dataset):
#     def __init__(self, params,mode):

#         self.input_size=int(params['input_size'])
#         self.num_classes=params['num_classes']
#         self.mode=mode
#         dataset_name=params['dataset_name']

#         preprocessfunc={'voc':self.pre_process_dataset,'nus':self.pre_process_dataset}
#         getfunc={'voc':self.get_item,'nus':self.get_item}

#         preprocessfunc[dataset_name](params)

#         self.get_function=getfunc[dataset_name]
        

#     def pre_process_dataset(self,params):

#         labels_files=ast.literal_eval(params[self.mode+'_labels_files'])
#         images_dirs=ast.literal_eval(params[self.mode+'_images_folders'])
        
#         #self.all_images=images_dirs[0]+self.img_labels.iloc[:,-1].to_numpy()
#         self.all_img_ids=np.empty(shape=(0,))
#         self.all_labels=np.empty(shape=(0,self.num_classes),dtype=np.float32)
#         self.all_images=np.empty(shape=(0,))

#         for i in range(len(labels_files)):
#             img_labels=pd.read_hdf(labels_files[i],key='df',mode='r')
#             self.all_img_ids=np.concatenate((self.all_img_ids,img_labels.iloc[:,-1].to_numpy()),axis=0)
#             self.all_labels=np.concatenate((self.all_labels,img_labels.iloc[:,:-1].to_numpy()),axis=0)
#             self.all_images=np.concatenate((self.all_images,images_dirs[i]+img_labels.iloc[:,-1].to_numpy()),axis=0)


#         #meanstds = get_pickle_data('/mnt/raptor/hassan/data/KG_data/voc/meanstd')

#         data_transforms = {
#             'train': transforms.Compose([
#                 transforms.RandomResizedCrop(self.input_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 #transforms.Normalize(meanstds['means'], meanstds['stds'])
#             ]),
#             'val': transforms.Compose([
#                 transforms.Resize(self.input_size),
#                 transforms.CenterCrop(self.input_size),
#                 transforms.ToTensor(),
#                 #transforms.Normalize(meanstds['means'], meanstds['stds'])
#             ]),
#             'test': transforms.Compose([
#                 transforms.Resize(self.input_size),
#                 transforms.CenterCrop(self.input_size),
#                 transforms.ToTensor(),
#                 #transforms.Normalize(meanstds['means'], meanstds['stds'])
#             ])
#         }
#         self.transform = data_transforms[self.mode]

#         self.selected_labels=self.all_labels
#         self.selected_images=self.all_images
#         self.selected_img_ids=self.all_img_ids
        
        
#     def set_class(self,c):
#         self.current_class=c
#         selected_indices=self.all_labels[:,c]==1
#         self.selected_labels=self.all_labels[select_indices,:]
#         print(self.selected_labels)
#         print(self.selected_labels.shape)
#         self.selected_images=self.all_images[selected_indices]
#         self.selected_img_ids=self.all_img_ids[select_indices]

#     def get_item(self,idx):
#         img_path=self.selected_images[idx]
#         image = Image.open(img_path).convert('RGB')
#         label = np.array(self.selected_labels[idx,:],dtype=np.float32)
#         return (self.transform(image),self.selected_img_ids[idx]),torch.from_numpy(label)        

#     # def get_item(self,idx):
#     #     img_path=self.all_images[idx]
#     #     image = Image.open(img_path).convert('RGB')
#     #     label = np.array(self.all_labels[idx,:],dtype=np.float32)
#     #     return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)

#     def __len__(self):
#         return len(self.all_images)

#     def __getitem__(self, idx):
#         return self.get_function(idx)



class ImageDataset(Dataset):
    def __init__(self, params,mode):

        self.input_size=int(params['input_size'])
        self.num_classes=int(params['num_classes'])
        self.mode=mode
        dataset_name=params['dataset_name']

        preprocessfunc={'voc':self.pre_process_dataset,'nus':self.pre_process_dataset}
        getfunc={'voc':self.get_item,'nus':self.get_item}

        preprocessfunc[dataset_name](params)

        self.get_function=getfunc[dataset_name]
        

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


        #meanstds = get_pickle_data('/mnt/raptor/hassan/data/KG_data/voc/meanstd')

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

        #selected_indices=np.zeros(self.all_labels.shape[0], dtype=bool)
                            
        # for ts in [1,2,3,4]:
        #     selected_indices=np.logical_or(selected_indices, self.all_labels[:,ts]==1.0)
        #print(list(np.sum(self.all_labels,axis=0)))
        self.all_labels=np.where(self.all_labels>0,1,0)
        #print(list(np.sum(self.all_labels,axis=0)))
        #print(list(np.sum(1-self.all_labels,axis=0)))
        #0/0
        print(self.all_labels.shape,self.all_images.shape,self.all_img_ids.shape)
        # self.all_labels=self.all_labels[selected_indices,:]
        # self.all_images=self.all_images[selected_indices,]
        # self.all_img_ids=self.all_img_ids[selected_indices,]
        # print('Selected:',self.all_labels.shape,self.all_images.shape,self.all_img_ids.shape)
        #self.select_label(0,0)
        
    def set_class(self,c):
        self.current_class=c
        selected_indices=self.all_labels[:,c]==1
        self.selected_labels=self.all_labels[select_indices,:]
        print(self.selected_labels)
        print(self.selected_labels.shape)
        self.selected_images=self.all_images[selected_indices]
        self.selected_img_ids=self.all_img_ids[select_indices]

    def get_item(self,idx):
        img_path=self.selected_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = np.array(self.selected_labels[idx,:],dtype=np.float32)
        return (self.transform(image),self.selected_img_ids[idx]),torch.from_numpy(label)        

    # def get_item(self,idx):
    #     img_path=self.all_images[idx]
    #     image = Image.open(img_path).convert('RGB')
    #     label = np.array(self.all_labels[idx,:],dtype=np.float32)
    #     return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)
    def select_label(self,labelidx,value):
        #selected_indices=np.where(self.all_labels[:,labelidx]==value)[0]
        #self.selected_labels=self.all_labels[selected_indices,:]
        #self.selected_images=self.all_images[selected_indices,]
        #self.selected_img_ids=self.all_img_ids[selected_indices,]

        self.selected_labels=self.all_labels
        self.selected_images=self.all_images
        self.selected_img_ids=self.all_img_ids
        print('Selected label and idx:',self.selected_labels.shape)


    def __len__(self):
        return len(self.all_images)
        #return len(self.selected_images)

    def __getitem__(self, idx):
        img_path=self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = np.array(self.all_labels[idx,:],dtype=np.float32)

        return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)

        #return self.get_function(idx)

