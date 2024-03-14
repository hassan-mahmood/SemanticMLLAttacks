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
from Attacks.apgd import *
import argparse
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from deepfool import * 
import string 
import shutil 
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

	def check_condition(self,present_to_absent,value):
		if(present_to_absent):
			if(value<0):
				return True 
		else:
			if(value>0):
				return True 
		return False 

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

		
		#print('Vector norm:',torch.linalg.vector_norm(self.Ux,ord=float('inf'),dim=2))
		#0/0
		class_num_for_grad=2
		present_to_absent=True
		use_deepfool=True

		store_folder='grad/'+('apgd','df')[use_deepfool]+'/c'+str(class_num_for_grad)+'/'+str((1,0)[present_to_absent])
		if(os.path.exists(store_folder)):
			shutil.rmtree(store_folder)
		create_folder(store_folder)
		self.logger.write('\nClass:'+str(class_num_for_grad)+', Present to absent:'+str(present_to_absent),', dir:',store_folder)

		current_target_value=(0.0,1.0)[present_to_absent]
		succount=0
		totcount=0

		apgdt = APGDAttack(model, eps=0.2, norm='Linf', n_iter=50, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))

		for epoch in range(start_epoch,start_epoch+num_epochs):
			#current_target_value=1.0-current_target_value

			avg_meter=AverageMeter()

			model.train()
			model.apply(self.set_bn_eval)
			model.model.eval()

			#torch.set_grad_enabled(True)
			self.logger.write('\nModel Training:')
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}

			for i, data in enumerate(tqdm(train_dataloader)):
				
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()

				with torch.no_grad():
					outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					

				#for jk in range(labels.shape[1]):
				for jk in range(class_num_for_grad,class_num_for_grad+1):
					t_class=torch.Tensor([jk]).long()
				#for jk in range(5):
					model.U.register_hook(lambda grad: torch.sign(grad) * 0.001)
					#select_indices=torch.where(torch.logical_and((labels[:,jk]==current_target_value),(labels[:,0]==1)))[0]
					select_indices=torch.where((labels[:,jk]==current_target_value))[0]
					
					if(select_indices.shape[0]==0):
						continue
					
					newlabels=torch.clone(labels[select_indices,:])
					
					inputs=all_inputs[select_indices,:,:,:] 
					train_optimizer.zero_grad()
					
					newlabels[:,jk] = 1 - newlabels[:,jk]

					#print(newlabels[:10,:5])

					#inputs = inputs + self.Ux[newlabels[:,t_class].long(),t_class,:]
					pert_image=torch.zeros_like(inputs)
					r_tot_all=torch.zeros_like(inputs)
					
					# out=self.Ux[newlabels[:,t_class].long(),t_class,:]
					# #print('out:',out.shape)
					
					# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
					# k=out*(epsilon_norm/(a+1e-10))[:,None]
					# input_images = inputs + k.view(k.shape[0],inputs.shape[1],inputs.shape[2],-1)
					# input_images=torch.clamp(input_images,min=0.0,max=1.0)
					# #print(torch.max(torch.abs(input_images)),torch.max(torch.abs(input_images-inputs)))
					input_images=inputs

					# print(newlabels[:,jk])
					#a=torch.from_numpy(np.load('first.npy'))
					
					
					
					
					############

					# #print(torch.linalg.vector_norm(a,ord=float('inf')))
					
					# #ux=0.2*torch.rand(ux.shape,dtype=torch.float32).cuda()
					
					# #48 508

					# #a=100*a
					
					# #b=np.load('second.npy').view(3,)
					# #print(a.shape,b.shape)
					# #r=np.matmul(a,b)
					# #

					# #print(torch.min(ux),torch.max(ux))

					# #for p in range(100):
					# for p in range(input_images.shape[0]):
					# 	x=input_images[p,:,:,:]+ux
					# 	#x=torch.rand_like(x)+ux
					# 	#x=torch.zeros_like(x)+ux
					# 	x=torch.clamp(x,min=0.0,max=1.0)

					# 	#print(torch.min(ux),torch.max(ux),torch.min(x),torch.max(x))
					# 	outputs,features = model(x,epsilon_norm=0.0,target_label=newlabels,target_class=t_class)
					# 	#print(outputs[:,jk])
					# 	for t in range(outputs.shape[0]):
					# 		if(outputs[t,jk]>0):
					# 			succount+=1

					# 		totcount+=1
					# 	#print('outputs:',outputs[:,jk])
					# 	# print('outputs:',outputs)
					# 	# 0/0
					# print(succount,totcount)
					# continue
					# print(ux.shape)
					# 0/0
					############
					

					

					if use_deepfool:
						r_tot, pert_image=deepfool(input_images, model, newlabels,t_class)
						0/0
						for p in range(input_images.shape[0]):
						#for p in range(10):
							r_tot, pert_image[p,:,:,:]=deepfool(input_images[p,:,:,:], model, newlabels,t_class)						
							r_tot_all[p,:,:,:]=torch.from_numpy(r_tot)
					else:
						selection_mask=torch.zeros_like(newlabels,dtype=torch.float32)
						selection_mask[:,jk]=1.0
						selection_mask=selection_mask.cpu().numpy()
						#r_tot, loop_i, pert_image=deepfool(inputs[0,:,:,:], model,jk)
						pert_image=apgdt.perturb(input_images,newlabels,criterion,use_target_indices=True,selection_mask=selection_mask)


					# #outputs,features = model(inputs[0,:,:,:],epsilon_norm=epsilon_norm,target_label=newlabels,target_class=t_class)
					
					u=torch.max(torch.abs(pert_image-input_images).view(input_images.shape[0],-1),dim=1)

					outputs,features = model(pert_image,epsilon_norm=0.0,target_label=newlabels,target_class=t_class)
					print(outputs[:,jk])
					for p in range(input_images.shape[0]):
						if self.check_condition(present_to_absent,outputs[p,jk]):
						# if(outputs[p,jk]<0):
							np.save(os.path.join(store_folder,''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))+'.npy'),(pert_image[p,:,:,:]-input_images[p,:,:,:]).flatten().detach().cpu().numpy())
					
					continue
					#print(outputs[:,jk])
					self.Ux[newlabels[:,t_class].long(),t_class,:] = self.Ux[newlabels[:,t_class].long(),t_class,:] + r_tot_all.view(r_tot_all.shape[0],1,-1)
					

					with torch.no_grad():
						#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
						#self.Ux.div_()
						normval=torch.linalg.vector_norm(self.Ux,ord=float('inf'),dim=2,keepdim=True)+1e-10
						self.Ux=self.Ux/normval


					totals[jk]+=outputs.shape[0]

					if(current_target_value==1.0):
						success[jk]+=torch.count_nonzero(outputs[:,jk]<=0).cpu()
					else:
						success[jk]+=torch.count_nonzero(outputs[:,jk]>0).cpu()
					#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
					#percents.append((torch.count_nonzero(outputs[:,jk]<=0)/outputs.shape[0]).item())
					#losses_dict,loss = criterion(outputs,newlabels,jk,model)
					#print(loss)
					#avg_meter.update('train_loss',loss.item())
					#loss.backward()
					# for p in model.parameters():
					# 	if(p.requires_grad is True):
					# 		#print(p.shape)
					# 		p.grad=0.0001*torch.sign(p.grad)
					#nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
					#train_optimizer.step()
					#scheduler.step()
					
					# with torch.no_grad():
					# 	model.Normalize_UAP()
				
				#print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]))
			# 	# print statistics
			print(succount,totcount)
			0/0
			trainstatout = '\nTrain - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
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
			#current_target_value=0.0
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
					for jk in range(num_classes):
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

						if(current_target_value==1.0):
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

