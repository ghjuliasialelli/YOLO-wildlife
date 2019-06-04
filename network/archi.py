import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from typing import Iterable, Tuple
from augment import datagen, Imgs, Labels

structure = [
		[3,64,3],
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,64,3],
		[64,3,3],
		]

class YOLO(nn.Module):
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''        
		super(YOLO, self).__init__()
		self.convs = []
		for x in structure:
			self.convs.append(nn.Conv2d(x[0],x[1],x[2],padding=(x[2]-1)//2))
		self.pool = nn.MaxPool2d(2)
	
	def forward(self,x):
		'''compute output of (batch size) * S * S * 3
		'''
		x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, epochs: int):
	'''given an initialized yolo object, train
	'''
	...



def datagen():
	while True:
		yield imgs, labels
