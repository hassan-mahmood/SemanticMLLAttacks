import sys
sys.path.append('./')
from scripts.tree_loss import *
from tqdm import tqdm 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


target_set_size=2


norm='inf'
datasetname='nus'

epsilons=np.linspace(0.001,0.005,5)
if datasetname=='voc':
	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
	#maindir='/mnt/raptor/hassan/aaai24/voc/OFFBetaOrthThresh/'
	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/vocconsistentON/','all'+str(target_set_size)),'rb'))['target_classes']
if datasetname =='nus':
	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/tree','rb'))
	maindir='/mnt/raptor/hassan/aaai24/nus/OFFBetaOrthThresh/'
	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
	# maindir='/mnt/raptor/hassan/aaai24/supp/nus/All/'
	# maindir='/mnt/raptor/hassan/cvpr24/supp/nus/All/'
	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/','all'+str(target_set_size)),'rb'))['target_classes']


epsilons=np.linspace(0.001,0.005,5)
print('All epsilons:',epsilons)
#epsilons=[0.009]
allsuccesses=[]
allconsistent=[]
allntr=[]

allsuc={}
allsucstd={}

allconsistent={}
allconsistentstd={}

allntr={}
allntrstd={}


for attacktype in ['mlabeta']:
	# acc=[]
	
	epschanged=[]
	epschangedstd=[]

	epssuc=[]
	epssucstd=[]

	epsconsistent=[]
	epsconsistentstd=[]


	print('Attack type:',attacktype)
	#for eps in [0.001,0.002,0.003,0.004,0.005]:
	for eps in epsilons:

		
		consistentpercent=[]
		successpercent=[]
		ntrpercent=[]

		eps=round(eps,3)
		print('Eps:',eps)

		for target_classes in alltargetlabels:
			classfolder='_'.join([str(k) for k in target_classes])

			current_path=os.path.join(maindir,str(eps),str(target_set_size),str(attacktype),classfolder)

			if not os.path.exists(os.path.join(current_path,'clean.h5')):
				# print('File does not exist')
				continue
			
			clean=pd.read_hdf(os.path.join(current_path,'clean.h5'),key='df').iloc[:,:-1].to_numpy()
			adv=pd.read_hdf(os.path.join(current_path,'adv.h5'),key='df').iloc[:,:-1].to_numpy()
			clean=np.where(clean>0,1,0)
			adv=np.where(adv>0,1,0)

			mytargetlabels=target_classes
			
			templabels=np.copy(adv)
			cleantemplabels=np.copy(clean)
		
			select_indices=np.ones_like(adv[:,0])
			#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
			mytargetlabels=target_classes

			for k in mytargetlabels:
				#print('Select indices:',k,select_indices.shape)
				select_indices=np.logical_and(select_indices==1.0,templabels[:,k]==0).astype(np.float32)

			success=np.count_nonzero(select_indices==1.0)/select_indices.shape[0]

			select_indices=np.where(select_indices==True)[0]
			
			totaltargetcount=0

			if attacktype == 'gmla':
				for c in select_indices:
					current_label=np.copy(cleantemplabels[c,:])
					treenodes=compute_tree_off_loss(np.copy(current_label),mytargetlabels,tree)
					totaltargetcount+=len(treenodes)

			else:
				totaltargetcount+=len(mytargetlabels)*len(select_indices)
			
			changed=(clean!=adv).astype(np.float32)
			ntrpercent.append((changed[select_indices].sum()-totaltargetcount)/(clean.shape[-1]*len(select_indices)-totaltargetcount+1))

			cleanconsistentcount=0
			local=0
			for c in select_indices:
				
				current_label=np.copy(templabels[c,:])
			
				targetlabels=mytargetlabels

				l=True 
				for k in targetlabels:
					templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
					l=l and templ

			
				#print('G:',g)
				if l is True:
					local+=1

			successpercent.append(len(select_indices)/templabels.shape[0])
			consistentpercent.append(local/templabels.shape[0])


		#print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])
		epschanged.append(np.mean(np.array(ntrpercent)))
		epschangedstd.append(np.std(np.array(ntrpercent)))

		epssuc.append(np.mean(np.array(successpercent)))
		epssucstd.append(np.std(np.array(successpercent)))

		epsconsistent.append(np.mean(np.array(consistentpercent)))
		epsconsistentstd.append(np.mean(np.array(consistentpercent)))
		# epsacc.append(np.mean(np.array(successpercent)))
		# epsconsistentacc.append(np.mean(np.array(consistentpercent)))
		# print(epsacc)
		# print(epsconsistentacc)

		
	#print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
	print('Attack:',attacktype,', success:',epssuc,', consistent:',epsconsistent,'ntr:',epschanged)
	# print(epschanged)
	allntr[attacktype]=epschanged
	allntrstd[attacktype]=epschangedstd

	allsuc[attacktype]=epssuc
	allsucstd[attacktype]=epssucstd

	allconsistent[attacktype]=epsconsistent
	allconsistentstd[attacktype]=epsconsistentstd 

	# consistentsuc[attacktype]=epschangedstd
	# suc[attacktype]=epsacc
	# consistentsuc[attacktype]=epsconsistentacc

	# allsuccesses.append(meansuc)
	# allconsistent.append(meanconsistent)

print('Success:')
print(allsuc)
print(allsucstd)

print('Consistent:')
print(allconsistent)
print(allconsistentstd)

print('All NTR:')
print(allntr)
print(allntrstd)
# print(suc)
# print(consistentsuc)
# print('Success:',allsuccesses)
# print('Consistent:',allconsistent)
# newsuc={}
# for k in suc.keys():
# 	newsuc[k]=suc[k]+[0.99,0.99,0.99,0.99,0.99]

0/0


# fig, ax = plt.subplots(figsize=(4,3.7))
# attacknames=[r'MLA-U',r'MLA-C',r'GMLA']
# attackindices=['mlaalpha','mlabeta','gmla']
# palette=plt.get_cmap('tab10')
# #markers=["*","v","o","d"]
# markers=["*","v","d"]
# linetypes=['-','-','-']
# epsilons=np.linspace(0.0,0.01,11)
# epsilonticks=np.linspace(0,0.01,6)
# # epsilons=np.linspace(0.0,0.005,6)
# # epsilonticks=np.linspace(0,0.005,6)
# # ax.set(xticks=pos, xticklabels=l)
# x=list(range(len(epsilons)))
# for attackidx in range(len(attacknames)):
#     ax.plot(x,[0]+newsuc[attackindices[attackidx]],label=attacknames[attackidx],marker=markers[attackidx],markersize=10,linewidth=3,color=palette(attackidx+1),linestyle=linetypes[attackidx])

# nodetext=''
# ax.set_ylabel(r'FR$_N$ '+('(One Node)','(Two Node)')[target_set_size-1],fontsize=18)
# ax.set_xlabel(r'$\epsilon$',fontsize=20)
# plt.yticks(fontsize=15)
# #plt.xticks(np.linspace(0,len(epsilons),len(epsilonticks)),[str(e) for e in epsilonticks],fontsize=12)
# plt.xticks(np.linspace(0,10,6),[str(e) for e in epsilonticks],fontsize=12)
# # ax.set(xticks=)

# plt.grid()
# leg=plt.legend(prop={'size':13},labelspacing=0.5,columnspacing=2.0,handletextpad=0.8,loc=3,fontsize=13,ncol=1,bbox_to_anchor=(0.55,0.08))
# ax.add_artist(leg)

# # lslabels=['One Optimization','Two Optimizations']
# # lstyles=['--','-']
# # h = [plt.plot([],[], color="black", marker="o", ls=lstyles[i],label=lslabels[i])[0] for i in range(2)]


# # # $plt.legend(handles=h, labels=lslabels,loc=(1.03,0.5), title="Quality")
# # plt.legend(handles=h, labels=lslabels,prop={'size':14},labelspacing=0.2,columnspacing=2.0,handletextpad=0.8,loc=3,fontsize=13,ncol=1,bbox_to_anchor=(0.01,0.26))
# plt.savefig('aaai24/voc/frn'+str(target_set_size)+'.pdf',dpi=150,bbox_inches='tight', pad_inches=0.1)




allsuccesses=[]
allconsistent=[]
for target_set_size in range(1,6):

	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/oiconsistentON/','new'+str(target_set_size)),'rb'))

	consistentpercent=[]
	successpercent=[]
	for current_target_labels in tqdm(alltargetlabels):
		
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/oiBetaOFF_graphonly/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		if not os.path.exists(current_store_folder):
			print(current_store_folder)
			continue
		if len(os.listdir(current_store_folder))==0:
			print(current_store_folder)
			continue
		clean=pd.read_hdf(os.path.join(current_store_folder,'clean.h5'),key='df').iloc[:,:-1].to_numpy()
		adv=pd.read_hdf(os.path.join(current_store_folder,'adv.h5'),key='df').iloc[:,:-1].to_numpy()
		clean=np.where(clean>0,1,0)
		adv=np.where(adv>0,1,0)
		mytargetlabels=current_target_labels
		
		templabels=np.copy(clean)
		
		#templabels[:,mytargetlabels]=(1 - target_on_off)	

		# cleanconsistentcount=0

		# for c in tqdm(range(adv.shape[0])):
		# 	current_label=np.zeros((601,),dtype=np.int32)
		# 	current_label[destindices]=np.copy(templabels[c,:][origindices])
		# 	_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
		# 	#print('G:',g)
		# 	if g is True:
		# 		cleanconsistentcount+=1

		# print(templabels.shape,cleanconsistentcount)


		templabels=np.copy(adv)
		
		select_indices=np.ones((adv.shape[0],),bool)
		#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
		for k in mytargetlabels:
			#print('Select indices:',k,select_indices.shape)
			select_indices=np.logical_and(select_indices,templabels[:,k]==1)

		success=np.count_nonzero(select_indices)/select_indices.shape[0]

		select_indices=np.where(select_indices==True)[0]
		#templabels[:,mytargetlabels]=(1 - target_on_off)	

		cleanconsistentcount=0
		local=0
		for c in select_indices:
			current_label=np.zeros((601,),dtype=np.int32)
			current_label[destindices]=np.copy(templabels[c,:][origindices])
			
			targetlabels=[mapping[t] for t in mytargetlabels if t in mapping.keys()]

			l=True 
			for k in targetlabels:
				templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
				l=l and templ

			
			#print('G:',g)
			if l is True:
				local+=1

		successpercent.append(len(select_indices)/templabels.shape[0])
		consistentpercent.append(local/templabels.shape[0])
		#print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])

	meansuc=np.mean(np.array(successpercent))
	meanconsistent=np.mean(np.array(consistentpercent))
	print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
	allsuccesses.append(meansuc)
	allconsistent.append(meanconsistent)

print('Success:',allsuccesses)
print('Consistent:',allconsistent)

0/0



labels=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabelsconsistent.h5',key='df').iloc[:,:-1].to_numpy()
labels=np.where(labels<=0,0,1)
alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oi/',str(target_set_size)),'rb'))
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))

mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))

origindices=list(mapping.keys())
destindices=list(mapping.values())


num_classes=tree.shape[0]

# consistent=0
# for c in tqdm(range(labels.shape[0])):
# 	current_label=np.zeros((601,),dtype=np.int32)
# 	current_label[destindices]=labels[c,:][origindices]
# 	_,g=check_local_global_consistency(current_label,0,num_classes,tree)
# 	if g is True:
# 		consistent+=1

# 	if consistent%100==0:
# 		print(consistent)

# print(labels.shape,consistent)
# 0/0


for mytargetlabels in alltargetlabels[:10]:
	consistent=0
	select_indices=np.ones((labels.shape[0],),dtype=bool)
	#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
	for k in mytargetlabels:
		#print('Select indices:',k,select_indices.shape)
		select_indices=np.logical_and(select_indices,labels[:,k]==target_on_off)

	print('Current select_indices',np.count_nonzero(select_indices))

	# current_labels=labels[select_indices,:]
	# current_labels=current_labels[:100]

	# current_labels[:,mytargetlabels]=1.0

	# for k in range(current_labels.shape[0]):
	# 	_,g=check_local_global_consistency(current_labels[k,:],0,num_classes,tree)
	# 	if g is True:
	# 		consistent+=1


	# print('Consistent:',consistent)

0/0
target_on_off=0
target_set_size=6

data=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabels3.h5',key='df')

labels=data.iloc[:,:-1].to_numpy()

labels=np.where(labels<=0,0,1)

imgids=data.iloc[:,-1].to_numpy().tolist()
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))

num_classes=tree.shape[0]
alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oi/',str(target_set_size)),'rb'))
mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))

origindices=list(mapping.keys())
destindices=list(mapping.values())

finalimgids=[]
for mytargetlabels in alltargetlabels[:10]:

	#select_indices=torch.ones(size=(labels.shape[0],)).type(torch.ByteTensor)
	select_indices=np.ones((labels.shape[0],),dtype=bool)
	#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
	for k in mytargetlabels:
		#print('Select indices:',k,select_indices.shape)
		select_indices=np.logical_and(select_indices,labels[:,k]==target_on_off)

	#print('Total indices:',np.count_nonzero(select_indices))
	final_indices=[]
	select_indices=np.where(select_indices==True)[0].tolist()

	for k in tqdm(range(labels.shape[0])):
		current_label=np.zeros((601,),dtype=np.int32)
		

		current_label[destindices]=labels[k,:][origindices]
		_,g=check_local_global_consistency(current_label,0,num_classes,tree)

		if g is True:
			final_indices.append(k)
			finalimgids.append(imgids[k])
	print('Total indices:',labels.shape[0],'Final indices:',len(final_indices))
	#0/0
	print('Image ids:',len(finalimgids))
	pickle.dump(finalimgids,open('/mnt/raptor/hassan/data/KG_data/oi/consistentimgids','wb'))
	0/0



