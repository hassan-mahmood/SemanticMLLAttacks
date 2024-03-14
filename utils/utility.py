

import numpy as np
import os
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from scipy.special import expit
from tqdm import tqdm 
from PIL import Image
import configparser
import ast, json
import argparse
import torch 
import math
from urllib.request import urlretrieve
import torch
import random
import torch.nn.functional as F
import numpy as np 

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)

np.set_printoptions(formatter={'float': '{0:0.3f}'.format})

def mtile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)
    
def rescale_to_image_range(inputs,dimval):
  with torch.no_grad():
    normval=torch.linalg.vector_norm(inputs,ord=float('inf'),dim=dimval,keepdim=True)+1e-10

  inputs=inputs/normval
  return inputs


def normalize_vec(U, max_norm, norm_p):

    if norm_p == 2:
        # out=torch.linalg.vector_norm(U,ord=float(norm_p),dim=1)
        # print('Inside',out,out.shape)
        # print('Max norm:',1,max_norm/out)
        # print(torch.min(1,torch.linalg.vector_norm(U,ord=float(norm_p),dim=1)))
        t=max_norm/torch.linalg.vector_norm(U,ord=float(norm_p),dim=1).cuda()
        t=torch.min(torch.ones_like(t).cuda(), t)
        print(t.shape,U.shape)
        U = U * t[:,None]
        0/0
        
        
    # elif norm_p == float('inf'):
        
    #     U = torch.sign(U) * torch.minimum(torch.abs(U), torch.ones_like(U)*max_norm)
    elif norm_p == float('inf'):
        
        U = torch.sign(U) * torch.minimum(torch.abs(U), torch.ones_like(U)*max_norm)
        #U = torch.sign(U) * torch.minimum(torch.abs(U), torch.einsum('ij,i->ij',torch.ones_like(U),max_norm))
        
    else:
         raise ValueError('Unknown norm value')

    return U

def normalize_and_scale(delta_im, mean_arr, std_arr,eps):

    delta_im = (delta_im + 1)*0.5

    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / std_arr[c]

    for i in range(delta_im.shape[0]):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = eps/std_arr[ci]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

def store_checkpoint(checkpointdict,checkpoint_path):
  #torch.save(checkpointdict, checkpoint_path,_use_new_zipfile_serialization=False)
  f=open('/proc/sys/kernel/hostname','r')
  if('pegasus' in f.read()):
    torch.save(checkpointdict, checkpoint_path)
  else:
    torch.save(checkpointdict, checkpoint_path,_use_new_zipfile_serialization=False)

def restore_checkpoint(checkpoint_path,model,optimizer,logger):
  
  checkpoint=torch.load(checkpoint_path)

  #print(checkpoint['model_state']['model.basemodel.7.2.bn3.weight'])
  #print(checkpoint['model_state'].keys())
  #0/0
  model.load_state_dict(checkpoint['model_state'])
  if(optimizer is not None):
    optimizer.load_state_dict(checkpoint['optim_state'])
  #optimizer.load_state_dict(checkpoint['optim_state'])
  #model_dict=checkpoint['model_state']
  epoch=checkpoint['epoch']
  min_val_loss=checkpoint['min_val_loss']
  current_val_loss=checkpoint['current_val_loss']
  training_loss=checkpoint['training_loss']
  start_epoch=epoch+1
  logger.write('Start Epoch:',start_epoch,', Min val loss:',min_val_loss,', Current val loss:',current_val_loss,', Training Loss:',training_loss)
  del checkpoint
  return start_epoch,min_val_loss


def eval_performance(raw_predictions,orig_labels,classnames):
  model_preds=np.where(raw_predictions>0,1,0)
  print('{0:^20} | {1:^10} | {2:^10} | {3:^10} | {4:^10} '.format('Class','Accuracy','Precision','Recall','F1score'))

  for c in range(raw_predictions.shape[1]):
      labels=orig_labels[:,c]
      preds=model_preds[:,c]
      print('{0:^20} | {1:^10.3f} | {2:^10.3f} | {3:^10.3f} | {4:^10.3f} '.format(classnames[c],accuracy_score(labels,preds),precision_score(labels,preds),recall_score(labels,preds),f1_score(labels,preds)))
  
  val_acc=accuracy_score(orig_labels,model_preds)
  #print(precision_recall_fscore_support(orig_labels,model_preds,average='macro'),precision_recall_fscore_support(orig_labels,model_preds,average='micro'),precision_recall_fscore_support(orig_labels,model_preds,average='weighted'))
  print('Exact Match Ratio:',val_acc)
  return val_acc


def compute_standard_mean_std(meanstdfile,all_images,percentimgs=0.5):
  0/0
  selected_images=np.random.choice(all_images.shape[0],size=(int(percentimgs*all_images.shape[0])),replace=False)
  means=[]
  stds=[]
  print('Number of images to find mean:',selected_images.shape)
  for idx in tqdm(selected_images):
      im=Image.open(all_images[idx]).convert('RGB')
      im=np.asarray(im, dtype=np.float32) / 255.0
      mean=np.mean(im,axis=(0,1))
      std=np.std(im,axis=(0,1))
      means.append(mean.tolist())
      stds.append(std.tolist())
  
  
  means=np.array(means)
  stds=np.array(stds)
  #print(means,'\n',stds)
  means,stds=np.mean(means,axis=0),np.mean(stds,axis=0)
  print(means,'\n',stds)
  store_pickle_data({'means':means,'stds':stds},meanstdfile)


def create_labels_file(img_ids,current_annotations_folder,labels_file_path):
    #print(len(allimages))
    all_labels=np.zeros(shape=(len(img_ids),len(all_class_names)))
    
    all_ids=[]
    for idx,imgname in enumerate(tqdm(img_ids)):
        all_labels[idx,:]=get_label(os.path.join(current_annotations_folder,imgname+'.xml'))
        all_ids.append(imgname)
    
    df=pd.DataFrame(all_labels)
    df['ids']=np.array(all_ids)
    df.to_hdf(labels_file_path,key='df',mode='w')
    print('Label file created at',labels_file_path)

def get_label(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    output_label=np.zeros(shape=(len(all_class_names,)),dtype=np.float)

    list_with_all_boxes = []
    classnames=[]
    for boxes in root.iter('object'):
        c=boxes.find('name').text
        c=c.replace(' ','_')
        c=c.replace('/','_')
        classnames.append(c)

    for c in classnames:
        output_label[indices_dict[c]]=1.0
        allparents=get_all_parent_nodes(indices_dict[c])
        for p in allparents:
            output_label[p]=1.0

    return output_label


def create_folder(folderpath):
    if(not os.path.exists(folderpath)):
        os.makedirs(folderpath)

def Merge_dict(a,b):
  return {**a, **b}

def get_pickle_data(filepath):
  
  if(not os.path.exists(filepath)):
    return None
    
  f=open(filepath,'rb')
  data=pickle.load(f)
  f.close()
  return data


def store_pickle_data(data,filepath):
  f=open(filepath,'wb')
  pickle.dump(data,f)
  f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
      self.runningavgs=dict()
      self.counts=dict()
      self.allkeys=[]
      
    def update(self,key,val):
      if(key not in self.allkeys):
        self.allkeys.append(key)
        self.counts[key]=1
        self.runningavgs[key]=val

      else:
        self.counts[key]+=1
        count=self.counts[key]
        self.runningavgs[key]=(self.runningavgs[key]+val/(count-1))*(count-1)/count

    def get_values(self):
      return self.runningavgs

    def get_stats(self,epoch,writer):
      statout='\nEpoch: %d, '%(epoch)
      #writer.add_scalars('Loss',{'Train Loss':runningavgs['train_loss'],'Val Loss':runningavgs['val_loss']},epoch)
      for k in self.allkeys:
        statout=statout+str(k)+': %.3f, '%(self.runningavgs[k])
        #writer.add_scalars('abc',28.9,0)
        writer.add_scalar(str(k),self.runningavgs[k],epoch) 
        # writer.add_scalar('BCE Loss',bce_loss_meter.avg,epoch)
        # writer.add_scalar('Norm Loss',norm_loss_meter.avg,epoch)
        # writer.add_scalar('Parent Margin Loss',parent_margin_loss_meter.avg,epoch)
        # writer.add_scalar('Neg Margin Loss',neg_margin_loss_meter.avg,epoch)
        # writer.add_scalars('Parent Margin Distances',{'Mean of Mean':mmean,'Std of Means':smean,'Mean of Stds':mstd,'Std of Stds':sstd},epoch)
      #statout='\nEpoch: %d , Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)
      #self.runningavgs['statout']=statout
      #return statout
      return self.runningavgs['train_loss'],self.runningavgs['val_loss'],statout
      #return self.runningavgs['train_adv_loss'],self.runningavgs['val_adv_loss'],self.runningavgs['val_loss'],statout

