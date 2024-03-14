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
import sklearn 
import argparse
import pickle
from sklearn.metrics import accuracy_score
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model

pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')

def load_model_weights(model, model_path):
	
	
	countyes=0
	countno=0
	countall=0
	state = torch.load(model_path, map_location='cpu')
	# for key in model.state_dict():
	# 	if key=='head.fc.weight' or key=='head.fc.bias':
	# 		print(key,'Counted')
	# 	print(key)


	for key in model.state_dict():
		countall+=1
		if 'num_batches_tracked' in key or 'head.fc' in key:
			countno+=1
			continue
		p = model.state_dict()[key]
		if key in state['state_dict']:
			ip = state['state_dict'][key]
			if p.shape == ip.shape:
				countyes+=1
				p.data.copy_(ip.data)  # Copy the data of parameters
			else:
				countno+=1
				print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
		else:
			countno+=1
			print('could not load layer: {}, not in checkpoint'.format(key))

	#746 642 104
	
	return model

	


class Trainer:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'train'})		
		#self.parse_data(configfile,experiment_name,losstype)

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

		batch_size=int(params['batch_size'])
		
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
		nowtime=datetime.now()

		args={'image_size':224,'model_name':'tresnet_l',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1/model-81.pt',
		'sep_features':0,
		#'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		'model_path':'/mnt/raptor/hassan/weights/lvis/asl/tresnet_l_v2_miil_21k.pth',
		'num_classes':1203,'workers':0
		}
		args=argparse.Namespace(**args)

		state = torch.load(args.model_path, map_location='cpu')

		#classes_list = np.array(list(state['idx_to_class'].values()))
		

		#args.num_classes = state['num_classes']
		args.num_classes=1203
		args.do_bottleneck_head = False
		model = create_model(args).cuda()

		model=load_model_weights(model, args.model_path)
		# tooptimparams=[]
		# for name, param in model.named_parameters():

		# 	if 'head.fc' in name:
		# 		tooptimparams.append(param)
		# 		param.requires_grad=True
		# 		# print(name, param.requires_grad)
		# 	else:
		# 		param.requires_grad=False 
		
		##############################

		# args={'image_size':224,'model_name':'tresnet_l',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1/model-81.pt',
		# 'sep_features':0,
		# #'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 'model_path':'/mnt/raptor/hassan/weights/lvis/asl/backupmodel-20.pt',
		# 'num_classes':1203,'workers':0
		# }
		# args=argparse.Namespace(**args)
		# state = torch.load(args.model_path, map_location='cpu')
		# args.num_classes=1203
		# args.do_bottleneck_head = False
		# model = create_model(args).cuda()
		# model.load_state_dict(state['model_state'], strict=True)

		start_epoch=51
		##############################

		# 0/0
		
		# print(state['state_dict'].keys())
		# model.load_state_dict(state['state_dict'], strict=True)

		# model.load_state_dict(state['model_state'], strict=True)
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		model=model.cuda()
		train_optimizer = optim.Adam(model.parameters(), lr=0.0001)
		#train_optimizer = optim.Adam(tooptimparams, lr=0.001)
		# scheduler = MultiStepLR(train_optimizer, milestones=[10,30,50], gamma=0.1)
		scheduler = MultiStepLR(train_optimizer, milestones=[1,10,30], gamma=0.1)

		for epoch in range(start_epoch,start_epoch+num_epochs):

			avg_meter=AverageMeter()

			model.train()

			for name, param in model.named_parameters():
				if 'head.fc' in name:
					assert (param.requires_grad is True)
					# print(name, param.requires_grad)
				else:
					assert (param.requires_grad is True)

			#torch.set_grad_enabled(True)
			self.logger.write('\nModel Training:')
			for i, data in enumerate(tqdm(train_dataloader)):
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				(inputs,img_ids),labels=data
				
				inputs=inputs.to(device)
				labels=labels.to(device).float()

				train_optimizer.zero_grad()
				outputs,features = model(inputs,[])
				loss = criterion(outputs,labels).sum()
				loss.backward()
				train_optimizer.step()
				
				if (i+1)%10==0:
					print('Loss:',loss.item())
				avg_meter.update('train_loss',loss.item())
				

			scheduler.step()
			model.eval()

			#torch.set_grad_enabled(False)
			self.logger.write('Model Evaluation:')

			vallabels=np.zeros(shape=(len(val_dataset),int(num_classes)),dtype=np.float32)
			valpreds=np.zeros(shape=(len(val_dataset),int(num_classes)),dtype=np.float32)
			print('Val shape:',vallabels.shape,valpreds.shape)
			dataidx=0
			with torch.no_grad():
				for i,data in enumerate(tqdm(val_dataloader)):
					#img_ids,inputs, labels = data 
					(inputs,img_ids),labels=data
					inputs=inputs.to(device)
					labels=labels.to(device).float()

					outputs,features  = model(inputs,[])
					
					vallabels[dataidx:dataidx+inputs.shape[0]]=labels.cpu()
					valpreds[dataidx:dataidx+inputs.shape[0]]=outputs.detach().cpu().float()
					
					dataidx=dataidx+inputs.shape[0]
					
					loss = criterion(outputs,labels).sum()
					avg_meter.update('val_loss',loss.item())

					#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			
			now = datetime.now()
			
			#val_acc=eval_performance(valpreds,vallabels,all_class_names)
			statout=now.strftime("%d/%m/%Y %H:%M:%S")
			out=avg_meter.get_stats(epoch,writer)
			train_loss,val_loss,newstatout=out
			valpreds=np.where(valpreds>0,1,0)
			vallabels=np.where(vallabels>0,1,0)
			val_acc=accuracy_score(vallabels,valpreds)
			writer.add_scalar('Val Accuracy',val_acc,epoch)

			f1score=sklearn.metrics.f1_score(vallabels,valpreds,average='weighted')
			
			
			statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)+', F1: %.3f'%(f1score)+', F1 nonweighted: %.3f'%(sklearn.metrics.f1_score(vallabels,valpreds,average='macro'))
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
			# print('Now time:',nowtime)
			
			print('Storing model')
			store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))




	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname)
trainer.train()

