import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from typing import Iterable, Tuple
from augment import datagen, Imgs, Labels

structure = [
		[
		[3,64,3], #Conv2d layers (input,output,kernel size)
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,3,3],
		],
		[2,4,6] #layers at which to do a Maxpool2d(of 2) pooling
		]


class YOLO(nn.Module):
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''        
		super(YOLO, self).__init__()
		self.convs = []
		for x in structure[0]:
			self.convs.append(nn.Conv2d(x[0],x[1],x[2],padding=(x[2]-1)//2))
		self.pool = nn.MaxPool2d(2)
		self.pools = structure[1]
	
	def forward(self,x):
		'''compute output of (batch size) * S * S * 3
		'''
		for i,conv in enumerate(self.convs):
			if i in self.pools:
				x = self.pool(x)
			x = F.relu(conv(x))
		return x

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, epochs: int):
	'''given an initialized yolo object, train
	'''
	...



def datagen():
	while True:
		yield imgs, labels
