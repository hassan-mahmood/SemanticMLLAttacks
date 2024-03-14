import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *
import pandas as pd 
from torchvision import transforms



# object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
#                      'bottle', 'bus', 'car', 'cat', 'chair',
#                      'cow', 'diningtable', 'dog', 'horse',
#                      'motorbike', 'person', 'pottedplant',
#                      'sheep', 'sofa', 'train', 'tvmonitor']
#object_categories=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/node_names','rb'))
object_categories=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/node_names','rb'))

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/voc/voc2012/vocdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/voc/voc2007/voctrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/voc/voc2007/voctest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/voc/voc2007/voctestnoimgs_06-Nov-2007.tar',
}



def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'vocdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)
    #num_classes=116

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'vocdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    path_devkit = os.path.join(root, 'vocdevkit')
    path_images = os.path.join(root, 'vocdevkit', 'voc2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
            util.download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
            util.download_url(urls['trainval_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'voc2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
            util.download_url(urls['test_images_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'voc2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
            util.download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


class voc2007Classification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'vocdevkit')
        self.path_images = os.path.join(root, 'vocdevkit', 'voc2007', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.input_size=224
        # download dataset
        download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'voc2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'voc2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        #self.classes = object_categories
        classnames_path='/mnt/raptor/hassan/data/KG_data/voc/node_names'
        #classnames_path='/mnt/raptor/hassan/data/KG_data/nus/node_names'
        self.all_class_names=pickle.load(open(classnames_path,'rb'))
        self.classes=self.all_class_names

        self.images = read_object_labels_csv(file_csv)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] voc 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))


        tree_path='/mnt/raptor/hassan/data/KG_data/voc/tree'
        
        self.tree=pickle.load(open(tree_path,'rb'))
        
        #self.images_folders=['/mnt/raptor/hassan/data/voc/']

        #self.train_labels_files=['/mnt/raptor/hassan/data/voc/Labels/newtrainlabels.h5']

        #self.val_labels_files=['/mnt/raptor/hassan/data/voc/Labels/newvallabels.h5']


        self.images_folders=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/JPEGImages/','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/']

        self.train_labels_files=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/train_labels.h5','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/train_labels.h5']

        self.val_labels_files=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/val_labels.h5','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/val_labels.h5']
        self.dataset_name='voc'
        self.meanstdfile='/mnt/raptor/hassan/data/KG_data/voc/meanstd'

        if(set=='trainval'):
            self.pre_process_dataset('train',self.train_labels_files,self.images_folders)
        if(set=='test'):
            self.pre_process_dataset('val',self.val_labels_files,self.images_folders)

        self.sampled_images=self.all_images
        self.sampled_labels=self.all_labels
        self.sampled_img_ids=self.all_img_ids

    def pre_process_dataset(self,mode,labels_files,images_dirs):

        #self.all_images=images_dirs[0]+self.img_labels.iloc[:,-1].to_numpy()
        self.all_img_ids=np.empty(shape=(0,))
        self.all_labels=np.empty(shape=(0,len(self.all_class_names)),dtype=np.float32)
        self.all_images=np.empty(shape=(0,))

        for i in range(len(labels_files)):
            img_labels=pd.read_hdf(labels_files[i],key='df',mode='r')
            self.all_img_ids=np.concatenate((self.all_img_ids,img_labels.iloc[:,-1].to_numpy()),axis=0)
            self.all_labels=np.concatenate((self.all_labels,img_labels.iloc[:,:-1].to_numpy()),axis=0)
            self.all_images=np.concatenate((self.all_images,images_dirs[i]+img_labels.iloc[:,-1].to_numpy()),axis=0)

        if((not os.path.exists(self.meanstdfile)) and mode=='train'):
            compute_standard_mean_std(self.meanstdfile,self.all_images,percentimgs=1.0)

        meanstds=pickle.load(open(self.meanstdfile,'rb'))

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
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
            ])
        }
        self.transform = data_transforms[mode]



    def __getitem__(self, idx):
        img_path=self.sampled_images[idx]

        #image = read_image(img_path)
        #image=io.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        #label = np.array(self.all_labels[idx,:],dtype=np.float32)
        label = np.array(self.sampled_labels[idx,:],dtype=np.float32)
        #return self.sampled_img_ids[idx],self.transform(image),torch.from_numpy(label)
        return (self.transform(image),self.sampled_img_ids[idx],self.inp),torch.from_numpy(label)

        # path, target = self.images[index]
        # print(path,target)
        # 0/0
        # img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return (img, path, self.inp), target

    def __len__(self):
        #return len(self.images)
        return len(self.sampled_images)

    def get_number_classes(self):
        return len(self.classes)
