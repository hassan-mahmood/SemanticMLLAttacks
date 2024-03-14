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
		model.Normalize_UAP()

		for epoch in range(start_epoch,start_epoch+num_epochs):

			avg_meter=AverageMeter()

			model.train()
			model.apply(self.set_bn_eval)
			model.model.eval()
			#torch.set_grad_enabled(True)
			self.logger.write('\nModel Training:')
			for i, data in enumerate(tqdm(train_dataloader)):
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				(inputs,img_ids),labels=data
				
				inputs=inputs.to(device)
				labels=labels.to(device).float()

				with torch.no_grad():
					outputs,features = model(inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
				

				#for jk in range(labels.shape[1]):
				for jk in range(labels.shape[1]):
					
					#inputs.requires_grad=True
					newlabels=torch.clone(labels)

					for _ in range(2):
						train_optimizer.zero_grad()
						newlabels[:,jk] = 1 - newlabels[:,jk]
						outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.Tensor([jk]).long())
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))

						losses_dict,loss = criterion(outputs,newlabels,jk,model)
						#print(loss)
						avg_meter.update('train_loss',loss.item())
						loss.backward()
						#nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
						train_optimizer.step()
						scheduler.step()
						with torch.no_grad():
							model.Normalize_UAP()
				
				#print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				
			# 	# print statistics

			model.eval()

			#torch.set_grad_enabled(False)
			self.logger.write('Model Evaluation:')
			vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			dataidx=0
			with torch.no_grad():
				for i,data in enumerate(tqdm(val_dataloader)):
					#img_ids,inputs, labels = data 
					(inputs,img_ids),labels=data
					inputs=inputs.to(device)
					labels=labels.to(device).float()

					outputs,features = model(inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					
					#for jk in range(labels.shape[1]):
					for jk in range(labels.shape[1]):

						newlabels=torch.clone(labels)
						#
						newlabels = 1 - newlabels
						newlabels[:,jk] = 1 - newlabels[:,jk]
					
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=jk)
						outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
						#print(torch.where(outputs[0,:]>0,1,0))
						vallabels[dataidx:dataidx+inputs.shape[0]]=newlabels.cpu()
						valpreds[dataidx:dataidx+inputs.shape[0]]=outputs.detach().cpu().float()
					
						dataidx=dataidx+inputs.shape[0]
					
						losses_dict,loss = criterion(outputs,newlabels,jk,model)
						avg_meter.update('val_loss',loss.item())

					#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			
			now = datetime.now()
			
			#val_acc=eval_performance(valpreds,vallabels,all_class_names)
			statout=now.strftime("%d/%m/%Y %H:%M:%S")
			train_loss,val_loss,newstatout=avg_meter.get_stats(epoch,writer)
			valpreds=np.where(valpreds>0,1,0)
			val_acc=accuracy_score(vallabels,valpreds)
			writer.add_scalar('Val Accuracy',val_acc,epoch)

			
			statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)
			#statout='\nEpoch: %d , Train loss: writer.add_scalar('Val Accuracy',val_acc,epoch)%.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)

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

