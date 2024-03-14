import torch 
from utils.utility import *
from tqdm import tqdm 
import numpy as np
import random 

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)

class OrthMLLAttacks:
	def __init__(self):
		self.step_scale_factor=1#3/4.0
		self.checkpoints=[-1,10, 20, 30, 60, 70]
		self.osc_past_iterations=4

	def check_oscillation(self,iterations_loss,current_iteration):
		# if oscillation in the past 10 iterations is more than 40%, then change
		if current_iteration-self.osc_past_iterations<=0:
			return 

		start_idx=current_iteration-self.osc_past_iterations
		#cons_sub=(torch.sign(iterations_loss[start_idx:current_iteration-1]-iterations_loss[start_idx+1:current_iteration])+1)/2

		cons_sub=torch.mean((torch.sign(iterations_loss[start_idx:current_iteration-1]-iterations_loss[start_idx+1:current_iteration])+1)/2,dim=0)

		# if the values oscillate 50% of the time, the values do not change or oscillate from greater to less and less to greater, then reduce the step size
		indices=cons_sub<=0.5
		self.step_size[indices]=self.step_size[indices]*self.step_scale_factor
		return indices 


	def check_decr(self,iterations_loss,current_iteration):
		start_iteration=self.checkpoints[self.checkpoints.index(current_iteration)-1]+1
		end_iteration=current_iteration-1
		change_in_loss=1-iterations_loss[end_iteration]/(iterations_loss[start_iteration]+1e-10)

		# if the loss has decreased by more than 25% and there is no oscillation, we good, otherwise change. 
		#indices=change_in_loss<0.25
		indices=np.logical_or(change_in_loss<0.25,change_in_loss>0.8)
		self.step_size[indices]=self.step_size[indices]*self.step_scale_factor
		return indices

	def pgd_optimize_perturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size,criterion,out_iterations,in_iterations):
			

		#50, 130, 250, 
		#chkpt_percent=[0.2,0.3,0.4,0.6]

		#self.step_size=(eps_val/out_iterations)*torch.ones(size=(inputs.shape[0],),dtype=torch.float32).cuda()
		
		
		pert_param = torch.nn.Parameter(torch.zeros_like(inputs,dtype=torch.float32).cuda())
		pert_param.requires_grad=True


		train_optimizer = torch.optim.Adam([pert_param], lr=float(1e-5))#,weight_decay=1e-4)
		

		for it in tqdm(range(out_iterations)):
			train_optimizer.zero_grad()

			#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
			#pert=torch.clone(best_pert)
			#############################

			model.zero_grad()

			newinputs=torch.clone(inputs)+pert_param
			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
			prev=torch.clone(outputs)
			koutputs=torch.where(outputs>0,1,0).float()

			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

			if((it+1)%10==0):
				
				scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
				#scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				scores=(koutputs==newlabels).float().sum(1)
				print('Success:',len(success_indices),', Scores:',scores.detach().cpu(),', Loss:',torch.sum(lossval[:,mytargetlabels]),', Min max:',torch.min(pert_param).item(),torch.max(pert_param).item(),', Sum:',torch.sum(pert_param))
				print(torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=2))

			
			normloss=torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=2).sum()

			lossval=lossval.sum() + normloss.sum()
			lossval.backward()

			train_optimizer.step()
			
			with torch.no_grad():
				#tempnorm=torch.linalg.vector_norm(pert_param.view(pert_param.shape[0],-1),ord=2,dim=1)
				#pert_param.data=(pert_param.data.view(pert_param.shape[0],-1)/torch.max(eps_val*torch.ones_like(tempnorm),tempnorm)[:,None]).view(pert_param.shape[0],3,224,224)
				pert_param.data=torch.clamp(pert_param,0.0,1.0)


			#print(torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=2),torch.min(pert_param),torch.max(pert_param))

		return torch.tensor(pert_param).detach().cpu()

	def optimize_perturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size,criterion,out_iterations,in_iterations):
		
		#50, 130, 250, 
		#chkpt_percent=[0.2,0.3,0.4,0.6]

		#self.step_size=(eps_val/out_iterations)*torch.ones(size=(inputs.shape[0],),dtype=torch.float32).cuda()
		
		
		pert_param = torch.nn.Parameter(torch.zeros_like(inputs,dtype=torch.float32).cuda())
		pert_param.requires_grad=True


		#train_optimizer = torch.optim.Adam([pert_param], lr=float(1.0))#,weight_decay=1e-4)
		train_optimizer = torch.optim.SGD([pert_param], lr=float(1.0))#,weight_decay=1e-4)
		

		for it in tqdm(range(out_iterations)):
			train_optimizer.zero_grad()

			#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
			#pert=torch.clone(best_pert)
			#############################

			model.zero_grad()

			newinputs=torch.clone(inputs)+pert_param
			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
			#print(outputs[:,1])
			prev=torch.clone(outputs)
			koutputs=torch.where(outputs>0,1,0).float()


			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

			if((it+1)%10==0):
				print(outputs[:,1])
				scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
				#scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				# scores=(koutputs==newlabels).float().sum(1)
				# print('Success:',len(success_indices),', Scores:',scores.detach().cpu(),torch.count_nonzero(scores==20).detach().cpu().item(),', Loss:',torch.sum(lossval[:,mytargetlabels]).item(),', Min max:',torch.min(pert_param).item(),torch.max(pert_param).item(),', Sum:',torch.sum(pert_param).item())
				# print(torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=float('inf')))
				scores=(koutputs[:,nontargetlabels]==newlabels[:,nontargetlabels]).float().sum(1)
				nontargetsuccessindices=torch.where((scores==len(nontargetlabels))==True)[0]
				print('Success:',len(success_indices),'NT Success:',len(nontargetsuccessindices),', Scores:',scores.detach().cpu(),', Loss:',torch.sum(lossval[:,mytargetlabels]),', Min max:',torch.min(pert_param).item(),torch.max(pert_param).item(),', Sum:',torch.sum(pert_param))
				# scores=(koutputs==newlabels).float().sum(1)
				# print('Success:',len(success_indices),', Scores:',scores.detach().cpu(),', Loss:',torch.sum(lossval[:,mytargetlabels]),', Min max:',torch.min(pert_param).item(),torch.max(pert_param).item(),', Sum:',torch.sum(pert_param))
				print(torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=float('inf')))

		
			nontargetgrads=[]
			

			for idx_ts,ts in enumerate(nontargetlabels):

				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				gradval=torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=pert_param,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
			orignontargetgrads=torch.cat(nontargetgrads,-1)
			nontargetgrads=torch.qr(orignontargetgrads)[0]

			# print(nontargetgrads.shape)
			# 0/0
			##############################
			#print('loss value shape:',lossval[:,mytargetlabels].shape)
			mytargetgrads=[]
			gradval=torch.autograd.grad(outputs=lossval[:,mytargetlabels].sum(),inputs=pert_param,retain_graph=False)[0].view(newinputs.shape[0],3*224*224)
			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			#print(gradval.shape)

			#0/0


			# mytargetgrads=[]

			# for idx_ts,ts in enumerate(mytargetlabels):

			# 	gradval=torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=pert_param,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


			#normloss=torch.linalg.vector_norm(pert_param,dim=(1,2,3),ord=2)
			#gradval=torch.autograd.grad(outputs=0.0001*normloss.sum()/normloss.shape[0],inputs=pert_param,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
			#gradval=torch.autograd.grad(outputs=normloss.sum()/normloss.shape[0],inputs=pert_param,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)

			#mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))

			mytargetgrads=torch.cat(mytargetgrads,-1)

			#print(mytargetgrads.shape,orignontargetgrads.shape)
			#0/0
			#####################
			# tempa=torch.cat((mytargetgrads,orignontargetgrads),dim=2)[0,:,:].squeeze()
			# tempa=tempa/(torch.linalg.vector_norm(tempa,ord=2,dim=0)[None,:]+1e-9)
			
			# dotprods=torch.matmul(tempa.T,tempa)
			# #print(dotprods)
			# dotprods=dotprods[torch.triu_indices(tempa.shape[1],tempa.shape[1]).unbind()]
			############
			#print(dotprods)
			#0/0
			#indices=torch.topk(dotprods,k=10,largest=True)
			#print(indices)
			#print(indices.shape)
			#print(torch.topk(dotprods,k=tempa.shape[1],largest=True)[0].detach().cpu().numpy(),torch.topk(dotprods,k=tempa.shape[1],largest=False)[0].detach().cpu().numpy())
			#print(torch.topk(dotprods,k=tempa.shape[1],largest=True)[1].detach().cpu().numpy(),torch.topk(dotprods,k=tempa.shape[1],largest=False)[1].detach().cpu().numpy())


			#print(mytargetgrads.shape)
			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
			#print('After reshape:',mytargetgrads.shape)
			
			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
			#print(proj_target_grads.shape)
			#print(proj_target_grads.shape,step_size)
			proj_target_grads=step_size*(proj_target_grads.squeeze(-1)/torch.linalg.vector_norm(proj_target_grads.squeeze(-1),ord=float('inf'),dim=1)[:,None])
			
			#print(proj_target_grads.shape)

			#gradval=proj_target_grads.detach().view(inputs.shape[0],3,224,224)
			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads.squeeze(-1),ord=float*)
			pert_param.grad=proj_target_grads.detach().view(inputs.shape[0],3,224,224)

			train_optimizer.step()

			with torch.no_grad():
				#tempnorm=torch.linalg.vector_norm(pert_param.view(pert_param.shape[0],-1),ord=2,dim=1)
				#pert_param.data=(pert_param.data.view(pert_param.shape[0],-1)/torch.max(eps_val*torch.ones_like(tempnorm),tempnorm)[:,None]).view(pert_param.shape[0],3,224,224)
				#pert_param.data=torch.clamp(normalize_vec(torch.tensor(pert_param.data),max_norm=eps_val,norm_p=norm).squeeze(-1),0.0,1.0)
				pert_param.data=torch.clamp(normalize_vec(torch.tensor(pert_param.data),max_norm=eps_val,norm_p=norm).squeeze(-1),0.0,1.0)


		return torch.tensor(pert_param).detach().cpu()

	# def autoperturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size,criterion,out_iterations,in_iterations):
			

	# 	#50, 130, 250, 
	# 	#chkpt_percent=[0.2,0.3,0.4,0.6]

	# 	#self.step_size=(eps_val/out_iterations)*torch.ones(size=(inputs.shape[0],),dtype=torch.float32).cuda()
	# 	self.step_size=step_size*torch.ones(size=(inputs.shape[0],)).cuda()
	# 	eps_val=eps_val*torch.ones(size=(inputs.shape[0],)).cuda()

	# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 	best_local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 	#prev_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		
		
	# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 	final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()


	# 	self.best_global_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 	self.best_global_ac_score=-1*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 	self.best_global_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 	self.best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		
	# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
	# 	prev=torch.zeros_like(newlabels)

	# 	iterations_loss=torch.zeros(size=(out_iterations,inputs.shape[0]),dtype=torch.float32)

	# 	t = 2 * torch.rand(inputs.shape).cuda().detach() - 1
	# 	#inputs = inputs + eps_val * torch.ones_like(inputs).detach() * self.normalize(t)


	# 	for it in tqdm(range(out_iterations)):

	# 		if it in self.checkpoints:
	# 			indices=self.check_decr(iterations_loss,it)
				
	# 			if ((it+1)%(self.osc_past_iterations+2)==0):
	# 				indices=np.logical_or(indices,self.check_oscillation(iterations_loss,it))

	# 			final_pert[indices]=self.best_global_pert[indices].detach()

	# 			#print('Step sizes:',self.step_size)

	# 		# elif ((it+1)%(self.osc_past_iterations+2)==0):
	# 		# 	indices=self.check_oscillation(iterations_loss,it)
	# 		# 	#final_pert[indices]=self.best_global_pert[indices].detach()
	# 		# 	print('Step sizes:',self.step_size)

	# 		# if it in checkpoints:
	# 		# 	Check if the value

	# 		# for k in range(inputs.shape[0]):
	# 		# 	if best_score[k]>0:
	# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
	# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 		best_local_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 		best_local_ac_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 		best_local_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
	# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	# 		#pert=torch.clone(best_pert)
	# 		#############################

	# 		model.zero_grad()

	# 		#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
	# 		newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
	# 		#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
	# 		newinputs.requires_grad=True
	# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
	# 		prev=torch.clone(outputs)
	# 		koutputs=torch.where(outputs>0,1,0).float()
			
	# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
	# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
			
	# 		#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
			
	# 		#scores=(koutputs==newlabels).float().sum(1)
	# 		#print('best score for s',s,best_score[s])

	# 		#gradval=-1*torch.autograd.grad(outputs=outputs[0,0].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)[0]
	# 		#gradval=gradval/torch.linalg.vector_norm(gradval,dim=0,ord=2)
			
	# 		#print(torch.linalg.vector_norm(gradval,dim=0,ord=2))
			
	# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

	# 		# gradval2=-1*torch.autograd.grad(outputs=lossval[0,0].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)[0]
	# 		# gradval2=gradval2/torch.linalg.vector_norm(gradval2,dim=0,ord=2)
	# 		# print(torch.linalg.vector_norm(gradval2,dim=0,ord=2))

	# 		# print(torch.dot(gradval,gradval2))

	# 		# 0/0
	# 		iterations_loss[it]=torch.mean(lossval[:,mytargetlabels],dim=1).detach().cpu()

	# 		mytargetgrads=[]

		
	# 		nontargetgrads=[]
	# 		for idx_ts,ts in enumerate(nontargetlabels):

	# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
	# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
	# 		nontargetgrads=torch.cat(nontargetgrads,-1)
	# 		nontargetgrads=torch.qr(nontargetgrads)[0]

	# 		# print(nontargetgrads.shape)
	# 		# 0/0
	# 		##############################
	# 		mytargetgrads=[]
	# 		for idx_ts,ts in enumerate(mytargetlabels):

	# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
	# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


	# 		mytargetgrads=torch.cat(mytargetgrads,-1)

	# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)

	# 		#mytargetgrads=torch.sign(mytargetgrads)

	# 		#print(mytargetgrads.shape)
	# 		##############################
			
	# 		#mytargetgrads=mytargetgrads+torch.randn_like(nontargetgrads[:,:,0]).unsqueeze(dim=-1)
			
			

	# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

	# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
	# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
	# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

	# 		#print(mytargetgrads.shape)


	# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
	# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
	# 		proj_target_grads=proj_target_grads.squeeze()
	# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
	# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

	# 		with torch.no_grad():
	# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
	# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				
	# 			#proj_target_grads=torch.sign(proj_target_grads)

	# 			#local_pert=normalize_vec(proj_target_grads,max_norm=self.step_size,norm_p=norm).squeeze()
				
	# 			local_pert=torch.sign(proj_target_grads)*self.step_size[:,None]


	# 			#print('Local pert:',torch.min(local_pert),torch.max(local_pert))
	# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
	# 			# topindices=torch.topk(local_pert,k=1000,dim=1)[1]
	# 			# belowindices=torch.topk(-1*local_pert,k=100,dim=1)[1]
	# 			# indices=torch.cat((topindices,belowindices),dim=1)

	# 			# sparse_pert=torch.zeros_like(local_pert)
	# 			# #sparse_pert[]
	# 			# rows=torch.arange(indices.shape[0]).expand(200,indices.shape[0]).T
	# 			# cols=indices
	# 			# sparse_pert[rows,cols]=local_pert[rows,cols]	
	# 			# local_pert=sparse_pert

	# 		with torch.no_grad():
	# 			#global_pert=final_pert+local_pert
	# 			global_pert=final_pert
	# 		#############################


	# 		for sub_it in range(in_iterations):
			
	# 			model.zero_grad()

	# 			newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
	# 			#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
				
	# 			newinputs.requires_grad=True
	# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
	# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

	# 			koutputs=torch.where(outputs>0,1,0).float()
				
	# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
	# 			ac_scores=(koutputs==newlabels).float().sum(1)
	# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
	# 			#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
	# 			#scores=(koutputs==newlabels).float().sum(1)
	# 			#print('best score for s',s,best_score[s])

	# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

	# 			#for s_idx,s in enumerate(success_indices):

	# 			for s in range(inputs.shape[0]):
	# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
	# 				nontarget_loss=torch.sum(lossval[s,nontargetlabels])
	# 				#if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
	# 				if scores[s]>best_local_score[s] or (scores[s]==best_local_score[s] and temp_loss<best_local_loss[s]):
	# 				#if scores[s]>best_local_score[s] or (scores[s]==best_local_score[s] and (ac_scores[s]>best_local_ac_score[s] or (ac_scores[s]==best_local_ac_score[s] and temp_loss<best_local_loss[s]))):
	# 					best_local_score[s]=scores[s]
	# 					best_local_pert[s]=torch.clone(local_pert[s]).detach()
	# 					best_local_loss[s]=temp_loss.detach()
	# 					best_local_ac_score[s]=ac_scores[s]
	# 					#best_global_pert[s]=torch.clone(global_pert[s]+local_pert[s]).detach()
	# 				# elif temp_loss<best_loss[s]:
	# 				# 	best_loss[s]=temp_loss

	# 			#################

	# 			# mytargetgrads=[]
	# 			# for idx_ts,ts in enumerate(mytargetlabels):

	# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


	# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

	# 			# ##############################
	# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
	# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

	# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
	# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
	# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


	# 			# #################
				
	# 			# if sub_it != 0:
	# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
	# 			# 	lossval=lossval+1000*torch.square(out[:,None])
	# 			#scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				
	# 			print('Success:',len(success_indices),', Scores:',ac_scores.detach().cpu(),', Loss: %.5f'%(torch.sum(lossval[:,mytargetlabels]).item()),', Min Max: %.5f %.5f'%(torch.min(final_pert).item(),torch.max(final_pert).item()))
				
	# 			# nontargetgrads=[]
	# 			# for idx_ts,ts in enumerate(nontargetlabels):

	# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
	# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
				
	# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
	# 			# nontargetgrads=torch.qr(nontargetgrads)[0]


	# 			mytargetgrads=[]
	# 			for idx_ts,ts in enumerate(mytargetlabels):

	# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
	# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
	# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


	# 			mytargetgrads=torch.cat(mytargetgrads,-1)
	# 			##############################
	# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)

	# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
	# 			#print(mytargetgrads.shape)
	# 			#proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
	# 			#print(proj_target_grads.shape)
	# 			#proj_target_grads=proj_target_grads.squeeze()
	# 			#proj_target_grads=mytargetgrads.squeeze()

				

	# 			with torch.no_grad():
	# 				#proj_target_grads=normalize_vec(proj_target_grads,max_norm=self.step_size,norm_p=norm).squeeze()
	# 				#local_pert=local_pert+proj_target_grads
	# 				#local_pert=0.75*local_pert+0.25*0.01*proj_target_grads
	# 				#local_pert=local_pert+proj_target_grads

	# 				proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
	# 				proj_target_grads=proj_target_grads.squeeze()


	# 				local_pert=local_pert+0.02*proj_target_grads
					
	# 				# local_pert=local_pert.unsqueeze(dim=-1)
	# 				# local_pert=local_pert-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),local_pert))

	# 				# local_pert=local_pert.unsqueeze(dim=-1)
	# 				# local_pert=local_pert-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),local_pert))
	# 				local_pert=local_pert.squeeze()

	# 				local_pert=normalize_vec(local_pert,max_norm=self.step_size,norm_p=norm).squeeze()

				
	# 			#print('Sum:',torch.sum(self.e))
	# 			del newinputs 
	# 			del outputs 
	# 			del lossval 
	# 			pert.requires_grad=False 

	# 		#print(best_score)

	# 		with torch.no_grad():
	# 			#final_pert=torch.clone(best_global_pert)
	# 			#final_pert=torch.clone(global_pert)+best_local_pert
	# 			final_pert=torch.clone(global_pert)+best_local_pert
	# 			#final_pert=torch.clone(global_pert)+0.25*prev_pert+0.75*best_local_pert
	# 			final_pert=normalize_vec(final_pert,max_norm=eps_val,norm_p=norm).squeeze()
	# 			#prev_pert=torch.clone(best_local_pert).detach()
	# 		# with torch.no_grad():
	# 		# 	global_pert=torch.clone(global_pert+best_pert).detach()
	# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

	# 		for s in range(final_pert.shape[0]):
	# 			if best_local_score[s]>self.best_global_score[s] or (self.best_global_score[s]==best_local_score[s] and best_local_loss[s]<self.best_global_loss[s]):
	# 				#print('This is better at loss:',best_local_score[s])
	# 				self.best_global_score[s]=best_local_score[s]
	# 				self.best_global_ac_score[s]=best_local_ac_score[s]
	# 				self.best_global_pert[s]=torch.clone(final_pert[s]).detach()
	# 				self.best_global_loss[s]=torch.clone(best_local_loss[s]).detach()
	# 				#best_global_pert[s]=torch.clone(global_pert[s]+local_pert[s]).detach()

	# 	return final_pert 
	def clip(self, x,norm,eps_val):
		if norm==float(np.inf):
			return torch.clamp(x, -eps_val, eps_val)
		else:
			0/0



	def autoperturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size,criterion,out_iterations,in_iterations):
			

		#50, 130, 250, 
		#chkpt_percent=[0.2,0.3,0.4,0.6]

		#self.step_size=(eps_val/out_iterations)*torch.ones(size=(inputs.shape[0],),dtype=torch.float32).cuda()
		self.step_size=step_size*torch.ones(size=(inputs.shape[0],)).cuda()
		#eps_val=eps_val*torch.ones(size=(inputs.shape[0],)).cuda()

		
		
		#final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).uniform_(-eps_val, eps_val).cuda()


		final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()



		self.best_global_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
		self.best_global_ac_score=-1*torch.ones((inputs.shape[0]),dtype=torch.float32)
		self.best_global_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
		self.best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()

		iterations_loss=torch.zeros(size=(out_iterations,inputs.shape[0]),dtype=torch.float32)

		#t = 2 * torch.rand(inputs.shape).cuda().detach() - 1
		#inputs = inputs + eps_val * torch.ones_like(inputs).detach() * self.normalize(t)


		for it in tqdm(range(out_iterations)):
			# if it in self.checkpoints:
			# 	indices=self.check_decr(iterations_loss,it)
				
			# 	if ((it+1)%(self.osc_past_iterations+2)==0):
			# 		indices=np.logical_or(indices,self.check_oscillation(iterations_loss,it))

			# 	final_pert[indices]=self.best_global_pert[indices].detach()

				#print('Step sizes:',self.step_size)

			model.zero_grad()
			#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
			#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs.requires_grad=True

			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
			prev=torch.clone(outputs)
			koutputs=torch.where(outputs>0,1,0).float()
			
			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			ac_scores=(koutputs==newlabels).float().sum(1)
			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
			
			
			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

			#print('Success:',len(success_indices),', Scores:',ac_scores.detach().cpu(),', Loss: %.5f'%(torch.sum(lossval[:,mytargetlabels]).item()),', Min Max: %.5f %.5f'%(torch.min(final_pert).item(),torch.max(final_pert).item()))

			iterations_loss[it]=torch.mean(lossval[:,mytargetlabels],dim=1).detach().cpu()
		
			nontargetgrads=[]
			for idx_ts,ts in enumerate(nontargetlabels):
				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
			nontargetgrads=torch.cat(nontargetgrads,-1)
			nontargetgrads=torch.qr(nontargetgrads)[0]

			# print(nontargetgrads.shape)
			# 0/0
			##############################
			mytargetgrads=[]
			for idx_ts,ts in enumerate(mytargetlabels):

				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


			mytargetgrads=torch.cat(mytargetgrads,-1)

			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)


			#mytargetgrads=(torch.sign(mytargetgrads).squeeze()*self.step_size[:,None]).unsqueeze(-1)

			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
			proj_target_grads=proj_target_grads.squeeze()
			#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

			with torch.no_grad():
				#local_pert=torch.sign(proj_target_grads)*self.step_size[:,None]
				#local_pert=torch.sign(proj_target_grads)*self.step_size[:,None]
				

				local_pert=normalize_vec(proj_target_grads,max_norm=self.step_size,norm_p=norm).squeeze()

				# # The problem is that this vector might not lie in the same subspace as the original spanned by the non-target complement
				# final_pert=self.clip(final_pert+local_pert,norm,eps_val)

				# tempnorm=torch.linalg.vector_norm(proj_target_grads,ord=float('inf'),dim=1)
				# local_pert=proj_target_grads/tempnorm[:,None]#.squeeze()
				# local_pert=local_pert*self.step_size[:,None]
				# print(torch.linalg.vector_norm(local_pert,ord=float('inf'),dim=1))
				# The problem is that this vector might not lie in the same subspace as the original spanned by the non-target complement
				#final_pert=self.clip(final_pert+local_pert,norm,eps_val)
				
				#final_pert=final_pert+local_pert
				#tempnorm=torch.linalg.vector_norm(final_pert,ord=float('inf'),dim=1)
				#final_pert=final_pert/tempnorm[:,None]#.squeeze()
				#final_pert=final_pert*(torch.ones_like(tempnorm)*eps_val)[:,None]
				#final_pert=self.clip(final_pert+local_pert,norm,eps_val)

				final_pert=normalize_vec(final_pert+local_pert,max_norm=torch.ones_like(self.step_size)*eps_val,norm_p=norm).squeeze()

			

			for s in range(final_pert.shape[0]):
				temp_loss=torch.sum(lossval[s,mytargetlabels])
				if scores[s]>self.best_global_score[s] or (self.best_global_score[s]==scores[s] and temp_loss<self.best_global_loss[s]):
					#print('This is better at loss:',best_local_score[s])
					self.best_global_score[s]=scores[s]
					self.best_global_ac_score[s]=ac_scores[s]
					self.best_global_pert[s]=torch.clone(final_pert[s]).detach()
					self.best_global_loss[s]=torch.clone(temp_loss).detach()
					#best_global_pert[s]=torch.clone(global_pert[s]+local_pert[s]).detach()

		return final_pert 

	def UMLLperturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size,criterion,out_iterations,in_iterations):
			

		#50, 130, 250, 
		#chkpt_percent=[0.2,0.3,0.4,0.6]

		#self.step_size=(eps_val/out_iterations)*torch.ones(size=(inputs.shape[0],),dtype=torch.float32).cuda()
		self.step_size=step_size*torch.ones(size=(inputs.shape[0],)).cuda()
		#eps_val=eps_val*torch.ones(size=(inputs.shape[0],)).cuda()
		#t = 2 * torch.rand(inputs.shape).cuda().detach() - 1
		#inputs = inputs + eps_val * torch.ones_like(inputs).detach() * self.normalize(t)


		for it in tqdm(range(out_iterations)):
			# if it in self.checkpoints:
			# 	indices=self.check_decr(iterations_loss,it)
				
			# 	if ((it+1)%(self.osc_past_iterations+2)==0):
			# 		indices=np.logical_or(indices,self.check_oscillation(iterations_loss,it))

			# 	final_pert[indices]=self.best_global_pert[indices].detach()

				#print('Step sizes:',self.step_size)

			model.zero_grad()
			#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs=torch.clone(inputs)
			#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs.requires_grad=True

			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
			prev=torch.clone(outputs)
			koutputs=torch.where(outputs>0,1,0).float()
			
			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))
		
			nontargetgrads=[]
			for idx_ts,ts in enumerate(nontargetlabels):
				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
			nontargetgrads=torch.cat(nontargetgrads,-1)
			#nontargetgrads=nontargetgrads.reshape(-1,150528).T
			nontargetgrads=torch.qr(nontargetgrads)[0]
			
			# print(nontargetgrads.shape)
			# 0/0
			##############################
			mytargetgrads=[]
			for idx_ts,ts in enumerate(mytargetlabels):

				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


			mytargetgrads=torch.cat(mytargetgrads,-1)

			#mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
			print(mytargetgrads.shape)
			print(nontargetgrads.shape)
			#mytargetgrads=(torch.sign(mytargetgrads).squeeze()*self.step_size[:,None]).unsqueeze(-1)

			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			#80 x 19 x x 80 x x

			proj_target_grads=mytargetgrads-torch.matmul(nontargetgrads,torch.matmul(torch.transpose(nontargetgrads,1,2),mytargetgrads))
			proj_target_grads=proj_target_grads.squeeze()
			#print(proj_target_grads.shape)
			#proj_target_grads=torch.sum(proj_target_grads,dim=0)
			#print(proj_target_grads.shape)
			#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

			with torch.no_grad():
				#local_pert=torch.sign(proj_target_grads)*self.step_size[:,None]
				#local_pert=torch.sign(proj_target_grads)*self.step_size[:,None]
				

				local_pert=normalize_vec(proj_target_grads,max_norm=self.step_size,norm_p=norm).squeeze()

				# # The problem is that this vector might not lie in the same subspace as the original spanned by the non-target complement
				# final_pert=self.clip(final_pert+local_pert,norm,eps_val)

				# tempnorm=torch.linalg.vector_norm(proj_target_grads,ord=float('inf'),dim=1)
				# local_pert=proj_target_grads/tempnorm[:,None]#.squeeze()
				# local_pert=local_pert*self.step_size[:,None]
				# print(torch.linalg.vector_norm(local_pert,ord=float('inf'),dim=1))
				# The problem is that this vector might not lie in the same subspace as the original spanned by the non-target complement
				#final_pert=self.clip(final_pert+local_pert,norm,eps_val)
				
				#final_pert=final_pert+local_pert
				#tempnorm=torch.linalg.vector_norm(final_pert,ord=float('inf'),dim=1)
				#final_pert=final_pert/tempnorm[:,None]#.squeeze()
				#final_pert=final_pert*(torch.ones_like(tempnorm)*eps_val)[:,None]
				#final_pert=self.clip(final_pert+local_pert,norm,eps_val)

				final_pert=normalize_vec(final_pert+local_pert,max_norm=torch.ones_like(self.step_size)*eps_val,norm_p=norm).squeeze()

			

			for s in range(final_pert.shape[0]):
				temp_loss=torch.sum(lossval[s,mytargetlabels])
				if scores[s]>self.best_global_score[s] or (self.best_global_score[s]==scores[s] and temp_loss<self.best_global_loss[s]):
					#print('This is better at loss:',best_local_score[s])
					self.best_global_score[s]=scores[s]
					self.best_global_ac_score[s]=ac_scores[s]
					self.best_global_pert[s]=torch.clone(final_pert[s]).detach()
					self.best_global_loss[s]=torch.clone(temp_loss).detach()
					#best_global_pert[s]=torch.clone(global_pert[s]+local_pert[s]).detach()

		return final_pert 

	def perturb(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,norm,step_size, criterion,out_iterations,in_iterations):

		#step_size=eps_val*torch.ones(size=(inputs.shape[0],)).cuda()
		#step_size=step_size.cuda()
		print('Step size:',step_size)
		print('Epsilon:',eps_val)
		step_size=step_size*torch.ones(size=(inputs.shape[0],)).cuda()
		eps_val=eps_val*torch.ones(size=(inputs.shape[0],)).cuda()

		pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		
		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
		final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()


		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
		best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
		nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
		prev=torch.zeros_like(newlabels)

		for it in tqdm(range(out_iterations)):

			# for k in range(inputs.shape[0]):
			# 	if best_score[k]>0:
			# 		pert[k]=torch.clone(best_pert[k]).detach()
			local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
			best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
			#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
			#pert=torch.clone(best_pert)
			#############################

			model.zero_grad()

			#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
			#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			newinputs.requires_grad=True
			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
			prev=torch.clone(outputs)
			koutputs=torch.where(outputs>0,1,0).float()
			
			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
			
			#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
			
			#scores=(koutputs==newlabels).float().sum(1)
			#print('best score for s',s,best_score[s])

			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

			mytargetgrads=[]

		
			nontargetgrads=[]
			for idx_ts,ts in enumerate(nontargetlabels):

				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
			nontargetgrads=torch.cat(nontargetgrads,-1)
			nontargetgrads=torch.qr(nontargetgrads)[0]

			# print(nontargetgrads.shape)
			# 0/0

			mytargetgrads=[]
			for idx_ts,ts in enumerate(mytargetlabels):

				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


			mytargetgrads=torch.cat(mytargetgrads,-1)

			##############################
			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
			#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

			#print(mytargetgrads.shape)


			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
			#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
			proj_target_grads=proj_target_grads.squeeze()

			with torch.no_grad():
				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).sqeeze()
				#local_pert=normalize_vec(proj_target_grads,max_norm=step_size/float(out_iterations),norm_p=float('inf')).squeeze()
				local_pert=normalize_vec(proj_target_grads,max_norm=step_size,norm_p=norm).squeeze()
				#print('Local pert:',torch.min(local_pert),torch.max(local_pert))
				#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


			with torch.no_grad():
				global_pert=normalize_vec(final_pert+local_pert,max_norm=eps_val,norm_p=norm)

			#############################
			print('Before min max:',torch.min(final_pert),torch.max(final_pert),torch.min(local_pert),torch.max(local_pert))

			for sub_it in range(in_iterations):
			
				model.zero_grad()

				#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
				newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
				
				newinputs.requires_grad=True
				outputs,_ = model(newinputs,{'epsilon_norm':0.0})
				#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

				koutputs=torch.where(outputs>0,1,0).float()
				
				scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
				
				success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
				#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
				#scores=(koutputs==newlabels).float().sum(1)
				#print('best score for s',s,best_score[s])

				lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

				#for s_idx,s in enumerate(success_indices):

				for s in range(inputs.shape[0]):
					temp_loss=torch.sum(lossval[s,mytargetlabels])
					nontarget_loss=torch.sum(lossval[s,nontargetlabels])
					#if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
					if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
						best_score[s]=scores[s]
						best_global_pert[s]=torch.clone(global_pert[s])
						#best_pert[s]=torch.clone(pert[s])
						best_pert[s]=torch.clone(local_pert[s]).detach()
						best_loss[s]=temp_loss.detach()
					# elif temp_loss<best_loss[s]:
					# 	best_loss[s]=temp_loss

				#################

				# mytargetgrads=[]
				# for idx_ts,ts in enumerate(mytargetlabels):

				# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
				# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


				# mytargetgrads=torch.cat(mytargetgrads,-1)

				# ##############################
				# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
				# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

				# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
				# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
				# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


				# #################
				
				# if sub_it != 0:
				# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
				# 	lossval=lossval+1000*torch.square(out[:,None])
					
				#print('Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
				
				scores=(koutputs==newlabels).float().sum(1)
				print('Success:',len(success_indices),', Scores:',scores.detach(),', Loss:',torch.sum(lossval[:,mytargetlabels]),torch.min(global_pert).item(),torch.max(global_pert).item())

				nontargetgrads=[]
				for idx_ts,ts in enumerate(nontargetlabels):

					#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
					gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
					nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
				
				nontargetgrads=torch.cat(nontargetgrads,-1)
				nontargetgrads=torch.qr(nontargetgrads)[0]

				mytargetgrads=[]
				for idx_ts,ts in enumerate(mytargetlabels):

					gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
					#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
					mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


				mytargetgrads=torch.cat(mytargetgrads,-1)

				#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
				#a=a/torch.max(a,dim=1)[0][:,None]
				
				#mytargetgrads=mytargetgrads/a[:,None,:]



				##############################
				mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
				#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

				#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
				
				#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
				#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

				#print(mytargetgrads.shape)


				# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
				
				proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
				proj_target_grads=proj_target_grads.squeeze()

				# print('\n\n\n\n\nSqueeze')
				#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
				# print(proj_target_grads.shape)
				# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
				# 0/0


				with torch.no_grad():
					#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
					#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
					

					#proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()

					
					#local_pert = local_pert+proj_target_grads
					##local_pert=normalize_vec(local_pert,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
					#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

					#proj_target_grads=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
					
					# #proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
					# #local_pert = local_pert+proj_target_grads
					# local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


					#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()
					
					#local_pert=proj_target_grads

					proj_target_grads=normalize_vec(proj_target_grads,max_norm=step_size,norm_p=norm).squeeze()

					local_pert+=proj_target_grads

					local_pert=normalize_vec(local_pert,max_norm=step_size,norm_p=norm).squeeze()
					#print('Inside Local pert:',torch.min(local_pert),torch.max(local_pert))
					global_pert=normalize_vec(global_pert+local_pert,max_norm=eps_val,norm_p=norm).squeeze()
					# print(torch.min(global_pert),torch.max(global_pert))
					#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


				
				#print('Sum:',torch.sum(self.e))
				del newinputs 
				del outputs 
				del lossval 
				pert.requires_grad=False 

			#print(best_score)

			with torch.no_grad():
				final_pert=torch.clone(best_global_pert)
			# with torch.no_grad():
			# 	global_pert=torch.clone(global_pert+best_pert).detach()
			# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

		return final_pert

#######################################

# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()

# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)

# 	out_iterations=300


# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		#global_pert=local_pert
# 		#############################


# 		for sub_it in range(1):
		
# 			model.zero_grad()

# 			newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			#scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(pert[s]).detach()
# 					best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss
				
# 			print('Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)

# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)

# 			# print('\n\n\n\n\nSqueeze')
# 			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
# 			# print(proj_target_grads.shape)
# 			# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
# 			# 0/0


# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				

# 				# proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 				# local_pert = local_pert+proj_target_grads
				
# 				# #proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# #local_pert = local_pert+proj_target_grads
# 				# local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 				#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()

# 				#pert=normalize_vec(mytargetgrads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				global_pert=normalize_vec(global_pert+mytargetgrads.squeeze(),max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#print(torch.min(global_pert),torch.max(global_pert))
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 		print(best_score)

# 		# with torch.no_grad():
# 		# 	global_pert=torch.clone(global_pert+local_pert).detach()
# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return global_pert 

### using the optimizer
# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	pert_param = torch.nn.Parameter(torch.zeros_like(inputs,dtype=torch.float32).cuda())
# 	pert_param.requires_grad=True
# 	train_optimizer = torch.optim.Adam([pert_param], lr=float(1e-3))#,weight_decay=1e-4)

# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)

# 	target_selection_mask=torch.zeros_like(newlabels)
# 	target_selection_mask[:,mytargetlabels]=1.0

# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)

# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		model.zero_grad()

# 		#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs=torch.clone(inputs)
# 		newinputs.requires_grad=True
# 		#newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
# 		newinputs=newinputs+final_pert.view(inputs.shape[0],3,224,224)
# 		#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
# 		#newinputs.requires_grad=True
# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 		prev=torch.clone(outputs)
# 		koutputs=torch.where(outputs>0,1,0).float()
		
# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
		
# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
		
# 		#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
		
# 		#scores=(koutputs==newlabels).float().sum(1)
# 		#print('best score for s',s,best_score[s])

# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 		mytargetgrads=[]

	
# 		nontargetgrads=[]
# 		for idx_ts,ts in enumerate(nontargetlabels):

# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			gradval=torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
		
# 		nontargetgrads=torch.cat(nontargetgrads,-1)
# 		nontargetgrads=torch.qr(nontargetgrads)[0]

# 		# print(nontargetgrads.shape)
# 		# 0/0

# 		mytargetgrads=[]
# 		for idx_ts,ts in enumerate(mytargetlabels):

# 			gradval=torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 		mytargetgrads=torch.cat(mytargetgrads,-1)

# 		##############################
# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 		#print(mytargetgrads.shape)


# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
		
# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

# 		with torch.no_grad():
# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			local_pert=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			#print('Local pert:',torch.min(local_pert),torch.max(local_pert))
# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 		with torch.no_grad():
# 			global_pert=final_pert.view(inputs.shape[0],3,224,224)+local_pert.view(inputs.shape[0],3,224,224)

# 		#############################


# 		for sub_it in range(in_iterations):
		
# 			model.zero_grad()

# 			#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 			newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			#scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				nontarget_loss=torch.sum(lossval[s,nontargetlabels])
# 				#if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					best_global_pert[s]=torch.clone(global_pert[s].view(1,-1))
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(local_pert[s].view(1,-1)).detach()
# 					best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			print('Success:',len(success_indices),', Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
			
# 			# nontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
# 			# nontargetgrads=torch.qr(nontargetgrads)[0]

# 			# localnontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	localnontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# localnontargetgrads=torch.cat(localnontargetgrads,-1)
# 			# localnontargetgrads=torch.qr(localnontargetgrads)[0]

			

# 			mytargetgrads=[]

# 			gradval=torch.autograd.grad(outputs=torch.mul(lossval,target_selection_mask).sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
			
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))

# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)


# 			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			#a=a/torch.max(a,dim=1)[0][:,None]
			
# 			#mytargetgrads=mytargetgrads/a[:,None,:]

# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
			
# 			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 			#print(mytargetgrads.shape)


# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
# 			#proj_target_grads=mytargetgrads-torch.bmm(localnontargetgrads,torch.bmm(torch.transpose(localnontargetgrads,1,2),mytargetgrads))
# 			#print(proj_target_grads.shape)

# 			#proj_target_grads=proj_target_grads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),proj_target_grads))
# 			#print(proj_target_grads.shape)
# 			#0/0

# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 			proj_target_grads=proj_target_grads.squeeze()

# 			#pert_param.register_hook(lambda grad: torch.sign(grad))
# 			pert_param.grad=proj_target_grads.detach().view(inputs.shape[0],3,224,224)

# 			# print('\n\n\n\n\nSqueeze')
# 			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
# 			# print(proj_target_grads.shape)
# 			# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
# 			# 0/0

# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				

# 				#proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()

				
# 				#local_pert = local_pert+proj_target_grads
# 				##local_pert=normalize_vec(local_pert,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 				#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 				#proj_target_grads=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
				
# 				# #proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# #local_pert = local_pert+proj_target_grads
# 				# local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 				#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()
				
# 				#local_pert=proj_target_grads
# 				proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 				local_pert+=proj_target_grads

# 				local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#print('Inside Local pert:',torch.min(local_pert),torch.max(local_pert))
# 				global_pert=normalize_vec(global_pert.view(newinputs.shape[0],3,224,224)+local_pert.view(newinputs.shape[0],3,224,224),max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# print(torch.min(global_pert),torch.max(global_pert))
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				train_optimizer.step()
# 				print('Sum:',torch.sum(pert_param))

			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 


# 		#print(best_score)

# 		with torch.no_grad():
# 			#final_pert=torch.clone(best_global_pert)
# 			final_pert=pert_param

# 		# with torch.no_grad():
# 		# 	global_pert=torch.clone(global_pert+best_pert).detach()
# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return final_pert 

########## some modifications in final part
# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()


# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)


# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		model.zero_grad()

# 		#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
# 		#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs.requires_grad=True
# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 		prev=torch.clone(outputs)
# 		koutputs=torch.where(outputs>0,1,0).float()
		
# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
		
# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
		
# 		#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
		
# 		#scores=(koutputs==newlabels).float().sum(1)
# 		#print('best score for s',s,best_score[s])

# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 		mytargetgrads=[]

	
# 		nontargetgrads=[]
# 		for idx_ts,ts in enumerate(nontargetlabels):

# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
		
# 		nontargetgrads=torch.cat(nontargetgrads,-1)
# 		nontargetgrads=torch.qr(nontargetgrads)[0]

# 		# print(nontargetgrads.shape)
# 		# 0/0

# 		mytargetgrads=[]
# 		for idx_ts,ts in enumerate(mytargetlabels):

# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 		mytargetgrads=torch.cat(mytargetgrads,-1)

# 		##############################
# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 		#print(mytargetgrads.shape)


# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
		
# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

# 		with torch.no_grad():
# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			local_pert=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			#print('Local pert:',torch.min(local_pert),torch.max(local_pert))
# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 		with torch.no_grad():
# 			#global_pert=final_pert+local_pert
# 			global_pert=final_pert
# 		#############################


# 		for sub_it in range(in_iterations):
		
# 			model.zero_grad()

# 			newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 			#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			#scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				nontarget_loss=torch.sum(lossval[s,nontargetlabels])
# 				#if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					best_global_pert[s]=torch.clone(global_pert[s])
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(local_pert[s]).detach()
# 					best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			print('Success:',len(success_indices),', Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
			
# 			# nontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
# 			# nontargetgrads=torch.qr(nontargetgrads)[0]

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)
# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)

# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 			proj_target_grads=proj_target_grads.squeeze()

# 			with torch.no_grad():
				
# 				proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				local_pert+=proj_target_grads
# 				# local_pert=local_pert.unsqueeze(dim=-1)
# 				# local_pert=local_pert-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),local_pert))
# 				local_pert=local_pert.squeeze()

# 				local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 		#print(best_score)

# 		with torch.no_grad():
# 			#final_pert=torch.clone(best_global_pert)

# 			if it>40:
# 				final_pert=0.75*torch.clone(global_pert)+0.25*best_pert
# 			else:
# 				final_pert=torch.clone(global_pert)+best_pert
# 		# with torch.no_grad():
# 		# 	global_pert=torch.clone(global_pert+best_pert).detach()
# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return final_pert 

####### using final_pert as the best global ### final hassan
# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	final_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()


# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)

# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		model.zero_grad()

# 		#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs=torch.clone(inputs)+final_pert.view(inputs.shape[0],3,224,224)
# 		#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs.requires_grad=True
# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 		prev=torch.clone(outputs)
# 		koutputs=torch.where(outputs>0,1,0).float()
		
# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
		
# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
		
# 		#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
		
# 		#scores=(koutputs==newlabels).float().sum(1)
# 		#print('best score for s',s,best_score[s])

# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 		mytargetgrads=[]

	
# 		nontargetgrads=[]
# 		for idx_ts,ts in enumerate(nontargetlabels):

# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
		
# 		nontargetgrads=torch.cat(nontargetgrads,-1)
# 		nontargetgrads=torch.qr(nontargetgrads)[0]

# 		# print(nontargetgrads.shape)
# 		# 0/0

# 		mytargetgrads=[]
# 		for idx_ts,ts in enumerate(mytargetlabels):

# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 		mytargetgrads=torch.cat(mytargetgrads,-1)

# 		##############################
# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 		#print(mytargetgrads.shape)


# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
		
# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

# 		with torch.no_grad():
# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			local_pert=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			#print('Local pert:',torch.min(local_pert),torch.max(local_pert))
# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 		with torch.no_grad():
# 			global_pert=final_pert+local_pert
# 		#############################


# 		for sub_it in range(in_iterations):
		
# 			model.zero_grad()

# 			#newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 			newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
			
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			#print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			#scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				nontarget_loss=torch.sum(lossval[s,nontargetlabels])
# 				#if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					best_global_pert[s]=torch.clone(global_pert[s])
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(local_pert[s]).detach()
# 					best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			#print('Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
			
# 			# nontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
# 			# nontargetgrads=torch.qr(nontargetgrads)[0]

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)

# 			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			#a=a/torch.max(a,dim=1)[0][:,None]
			
# 			#mytargetgrads=mytargetgrads/a[:,None,:]



# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
			
# 			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 			#print(mytargetgrads.shape)


# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 			proj_target_grads=proj_target_grads.squeeze()

# 			# print('\n\n\n\n\nSqueeze')
# 			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
# 			# print(proj_target_grads.shape)
# 			# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
# 			# 0/0


# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				

# 				#proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()

				
# 				#local_pert = local_pert+proj_target_grads
# 				##local_pert=normalize_vec(local_pert,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 				#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 				#proj_target_grads=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
				
# 				# #proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# #local_pert = local_pert+proj_target_grads
# 				# local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 				#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()
				
# 				#local_pert=proj_target_grads
# 				proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 				local_pert+=proj_target_grads

# 				local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#print('Inside Local pert:',torch.min(local_pert),torch.max(local_pert))
# 				global_pert=normalize_vec(global_pert+local_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# print(torch.min(global_pert),torch.max(global_pert))
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 		#print(best_score)

# 		with torch.no_grad():
# 			final_pert=torch.clone(best_global_pert)
# 		# with torch.no_grad():
# 		# 	global_pert=torch.clone(global_pert+best_pert).detach()
# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return final_pert 

#################################################
# Changing global_pert with every iteration

# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()

# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)

# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		model.zero_grad()

# 		newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 		#newinputs=torch.clone(inputs)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs.requires_grad=True
# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 		prev=torch.clone(outputs)
# 		koutputs=torch.where(outputs>0,1,0).float()
		
# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
		
# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 		print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
		
# 		#scores=(koutputs==newlabels).float().sum(1)
# 		#print('best score for s',s,best_score[s])

# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 		mytargetgrads=[]

	
# 		nontargetgrads=[]
# 		for idx_ts,ts in enumerate(nontargetlabels):

# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
		
# 		nontargetgrads=torch.cat(nontargetgrads,-1)
# 		nontargetgrads=torch.qr(nontargetgrads)[0]

# 		# print(nontargetgrads.shape)
# 		# 0/0

# 		mytargetgrads=[]
# 		for idx_ts,ts in enumerate(mytargetlabels):

# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 		mytargetgrads=torch.cat(mytargetgrads,-1)

# 		##############################
# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 		#print(mytargetgrads.shape)


# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
		
# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

# 		with torch.no_grad():
# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			local_pert=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 		#global_pert=local_pert
# 		#############################


# 		for sub_it in range(in_iterations):
		
# 			model.zero_grad()

# 			newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
			
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			#scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(local_pert[s]).detach()
# 					best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			print('Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
			
# 			# nontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
# 			# nontargetgrads=torch.qr(nontargetgrads)[0]

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)

# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
			
# 			#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 			#print(mytargetgrads.shape)


# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 			proj_target_grads=proj_target_grads.squeeze()

# 			# print('\n\n\n\n\nSqueeze')
# 			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
# 			# print(proj_target_grads.shape)
# 			# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
# 			# 0/0


# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				

# 				# proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 				# local_pert = local_pert+proj_target_grads
				
# 				# #proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				# #local_pert = local_pert+proj_target_grads
# 				# local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


# 				#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()
				
# 				local_pert=proj_target_grads

# 				local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				global_pert=normalize_vec(global_pert+local_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				print(torch.min(global_pert),torch.max(global_pert))
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 		print(best_score)

# 		# with torch.no_grad():
# 		# 	global_pert=torch.clone(global_pert+local_pert).detach()
# 		# 	global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return global_pert 
#################################################

# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):

# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))

# 	for it in tqdm(range(out_iterations)):

# 		for k in range(inputs.shape[0]):
# 			if best_score[k]>0:
# 				pert[k]=torch.clone(best_pert[k]).detach()
# 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)

# 		for sub_it in range(in_iterations+1):
		
# 			model.zero_grad()

# 			newinputs=torch.clone(inputs)+pert.view(inputs.shape[0],3,224,224)
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			print('Success:',len(success_indices),torch.min(pert),torch.max(pert))
# 			#scores=(koutputs==newlabels).float().sum(1)
		
			

# 				#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			for s_idx,s in enumerate(success_indices):
# 				temp_loss=torch.sum(lossval[s,:])
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					best_pert[s]=torch.clone(pert[s])
# 					best_loss[s]=torch.sum(lossval[s,:])
# 				elif temp_loss<best_loss[s]:
# 					best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			print('Scores:',scores,', Loss:',torch.sum(lossval))
		
# 			mytargetgrads=[]

# 			if sub_it==0:
# 				nontargetgrads=[]
# 				for idx_ts,ts in enumerate(nontargetlabels):

# 					#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 					gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 					nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
				
# 				nontargetgrads=torch.cat(nontargetgrads,-1)
# 				nontargetgrads=torch.qr(nontargetgrads)[0]

# 			# print(nontargetgrads.shape)
# 			# 0/0

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)

# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 			#print(mytargetgrads.shape)


# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))


# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				pert = normalize_vec(pert+proj_target_grads.squeeze(),max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 	return pert 

###########################

#### works for correlation high ones.
# import torch 
# from utils.utility import *
# from tqdm import tqdm 
# def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):
# 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
	
# 	local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()

# 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))
# 	prev=torch.zeros_like(newlabels)


# 	global_best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 	global_best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 	global_best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)

# 	for it in tqdm(range(out_iterations)):

# 		# for k in range(inputs.shape[0]):
# 		# 	if best_score[k]>0:
# 		# 		pert[k]=torch.clone(best_pert[k]).detach()
# 		local_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# 		#pert=torch.clone(best_pert)
# 		#############################

# 		model.zero_grad()

# 		newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 		newinputs.requires_grad=True
# 		outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 		prev=torch.clone(outputs)
# 		koutputs=torch.where(outputs>0,1,0).float()
		
# 		scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
		
# 		success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 		print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
		
# 		#scores=(koutputs==newlabels).float().sum(1)
# 		#print('best score for s',s,best_score[s])

# 		lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 		mytargetgrads=[]

	
# 		nontargetgrads=[]
# 		for idx_ts,ts in enumerate(nontargetlabels):

# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
		
# 		nontargetgrads=torch.cat(nontargetgrads,-1)
# 		nontargetgrads=torch.qr(nontargetgrads)[0]

# 		# print(nontargetgrads.shape)
# 		# 0/0

# 		mytargetgrads=[]
# 		for idx_ts,ts in enumerate(mytargetlabels):

# 			gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 		mytargetgrads=torch.cat(mytargetgrads,-1)

# 		##############################
# 		mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 		#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 		#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 		#a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 		#mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 		#print(mytargetgrads.shape)


# 		# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
		
# 		proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 		#local_pert=torch.clone(proj_target_grads.squeeze()).detach()
# 		#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]

# 		with torch.no_grad():
# 			#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 			#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 			local_pert=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 			#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()



# 		#############################


# 		for sub_it in range(in_iterations+5):
		
# 			model.zero_grad()

# 			newinputs=torch.clone(inputs)+local_pert.view(inputs.shape[0],3,224,224)+global_pert.view(inputs.shape[0],3,224,224)
# 			newinputs.requires_grad=True
# 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 			#print('Prev diff:',torch.sum(torch.abs(outputs-prev)[:,mytargetlabels]),torch.sum(torch.abs(outputs-prev)[:,nontargetlabels]))

# 			koutputs=torch.where(outputs>0,1,0).float()
			
# 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# 			print('Success:',len(success_indices),torch.min(local_pert).item(),torch.max(local_pert).item(),torch.min(global_pert).item(),torch.max(global_pert).item())
# 			#scores=(koutputs==newlabels).float().sum(1)
# 			#print('best score for s',s,best_score[s])

# 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# 			#for s_idx,s in enumerate(success_indices):

# 			for s in range(inputs.shape[0]):
# 				temp_loss=torch.sum(lossval[s,mytargetlabels])
# 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# 					best_score[s]=scores[s]
# 					#best_pert[s]=torch.clone(pert[s])
# 					best_pert[s]=torch.clone(local_pert[s]).detach()
# 					best_loss[s]=temp_loss.detach()

# 				if scores[s]>global_best_score[s] or (scores[s]==global_best_score[s] and temp_loss<global_best_loss[s]):
# 					global_best_score[s]=scores[s]
# 					#best_pert[s]=torch.clone(pert[s])
# 					global_best_pert[s]=torch.clone(local_pert[s]+global_pert[s]).detach()
# 					global_best_loss[s]=temp_loss.detach()
# 				# elif temp_loss<best_loss[s]:
# 				# 	best_loss[s]=temp_loss

# 			#################

# 			# mytargetgrads=[]
# 			# for idx_ts,ts in enumerate(mytargetlabels):

# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# 			# ##############################
# 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# 			# #################
			
# 			# if sub_it != 0:
# 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# 			print('Scores:',scores,', Loss:',torch.sum(lossval[:,mytargetlabels]))
			
# 			# nontargetgrads=[]
# 			# for idx_ts,ts in enumerate(nontargetlabels):

# 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 			# 	nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
			
# 			# nontargetgrads=torch.cat(nontargetgrads,-1)
# 			# nontargetgrads=torch.qr(nontargetgrads)[0]

# 			mytargetgrads=[]
# 			for idx_ts,ts in enumerate(mytargetlabels):

# 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# 			mytargetgrads=torch.cat(mytargetgrads,-1)

# 			##############################
# 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
			
# 			a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# 			mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# 			#print(mytargetgrads.shape)


# 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))
# 			proj_target_grads=proj_target_grads.squeeze()

# 			# print('\n\n\n\n\nSqueeze')
# 			#proj_target_grads=proj_target_grads/torch.linalg.vector_norm(proj_target_grads,dim=1)[:,None]
# 			# print(proj_target_grads.shape)
# 			# print(torch.mm(proj_target_grads.squeeze(),proj_target_grads.squeeze().T))
# 			# 0/0


# 			with torch.no_grad():
# 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				

# 				proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations*in_iterations),norm_p=float('inf')).squeeze()
# 				local_pert = local_pert+proj_target_grads
				
# 				#proj_target_grads=normalize_vec(proj_target_grads,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#local_pert = local_pert+proj_target_grads
				
# 				local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
				
# 				#global_pert=normalize_vec(global_pert+proj_target_grads,max_norm=(sub_it+1)*eps_val/float(out_iterations*in_iterations)+(it*eps_val)/float(out_iterations),norm_p=float('inf')).squeeze()
				


# 				#local_pert=normalize_vec(local_pert,max_norm=eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#global_pert=normalize_vec(global_pert+local_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()
# 				#pert = normalize_vec(pert+proj_target_grads,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()


			
# 			#print('Sum:',torch.sum(self.e))
# 			del newinputs 
# 			del outputs 
# 			del lossval 
# 			pert.requires_grad=False 

# 		print(best_score)

# 		with torch.no_grad():
# 			global_pert=torch.clone(global_pert+best_pert).detach()
# 			global_pert=normalize_vec(global_pert,max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

# 	return global_best_pert 


# # def orth_mll_attack(model,inputs,newlabels,mytargetlabels,nontargetlabels, eps_val,criterion,out_iterations,in_iterations):

# # 	pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# # 	best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# # 	best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# # 	best_loss = 1000000*torch.ones((inputs.shape[0]),dtype=torch.float32)
# # 	nontargetgrads=torch.zeros((inputs.shape[0],3*224*224,len(nontargetlabels)))

# # 	for it in tqdm(range(out_iterations)):

# # 		for k in range(inputs.shape[0]):
# # 			if best_score[k]>0:
# # 				pert[k]=torch.clone(best_pert[k]).detach()
# # 		best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)
# # 		#best_pert = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
# # 		#pert=torch.clone(best_pert)

# # 		for sub_it in range(in_iterations+1):
		
# # 			model.zero_grad()

# # 			newinputs=torch.clone(inputs)+pert.view(inputs.shape[0],3,224,224)
# # 			newinputs.requires_grad=True
# # 			outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# # 			koutputs=torch.where(outputs>0,1,0).float()
			
# # 			scores=(koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)
			
# # 			success_indices=torch.where((scores==len(mytargetlabels))==True)[0]
# # 			print('Success:',len(success_indices),torch.min(pert),torch.max(pert))
# # 			#scores=(koutputs==newlabels).float().sum(1)
		
			

# # 				#print('best score for s',s,best_score[s])

# # 			lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))

# # 			for s_idx,s in enumerate(success_indices):
# # 				temp_loss=torch.sum(lossval[s,:])
# # 				if scores[s]>best_score[s] or (scores[s]==best_score[s] and temp_loss<best_loss[s]):
# # 					best_score[s]=scores[s]
# # 					best_pert[s]=torch.clone(pert[s])
# # 					best_loss[s]=torch.sum(lossval[s,:])
# # 				elif temp_loss<best_loss[s]:
# # 					best_loss[s]=temp_loss

# # 			#################

# # 			# mytargetgrads=[]
# # 			# for idx_ts,ts in enumerate(mytargetlabels):

# # 			# 	gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# # 			# 	#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# # 			# 	mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# # 			# mytargetgrads=torch.cat(mytargetgrads,-1)

# # 			# ##############################
# # 			# mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# # 			# #print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# # 			# #mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# # 			# a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# # 			# mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]


# # 			# #################
			
# # 			# if sub_it != 0:
# # 			# 	out= torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads).sum(dim=(1,2))
# # 			# 	lossval=lossval+1000*torch.square(out[:,None])
				
# # 			print('Scores:',scores,', Loss:',torch.sum(lossval))
		
# # 			mytargetgrads=[]

# # 			if sub_it==0:
# # 				nontargetgrads=[]
# # 				for idx_ts,ts in enumerate(nontargetlabels):

# # 					#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# # 					gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# # 					nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
				
# # 				nontargetgrads=torch.cat(nontargetgrads,-1)
# # 				nontargetgrads=torch.qr(nontargetgrads)[0]

# # 			# print(nontargetgrads.shape)
# # 			# 0/0

# # 			mytargetgrads=[]
# # 			for idx_ts,ts in enumerate(mytargetlabels):

# # 				gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
# # 				#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
# # 				mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))


# # 			mytargetgrads=torch.cat(mytargetgrads,-1)

# # 			##############################
# # 			mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
# # 			#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

# # 			#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
# # 			a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
# # 			mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

# # 			#print(mytargetgrads.shape)


# # 			# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
			
# # 			proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))


# # 			with torch.no_grad():
# # 				#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
# # 				pert = normalize_vec(pert+proj_target_grads.squeeze(),max_norm=(it+1)*eps_val/float(out_iterations),norm_p=float('inf')).squeeze()

			
# # 			#print('Sum:',torch.sum(self.e))
# # 			del newinputs 
# # 			del outputs 
# # 			del lossval 
# # 			pert.requires_grad=False 

# # 	return pert 
