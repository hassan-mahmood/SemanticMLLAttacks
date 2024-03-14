import torch
from torch.utils.tensorboard import SummaryWriter



class Logger:
	def __init__(self,filepath):
		self.filepath=filepath
		self.writetofile=True
		if(len(filepath)==0):
			self.writetofile=False
		

	def write(self,*data):
		data=' '.join(map(str,data))
		print(data)
		if(self.writetofile is True):
			with open(self.filepath,'a+') as f:
				f.write('\n'+data)
		
		# filepath: Path to the text file to store the log
		