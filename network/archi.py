import torch.nn as nn
import torch.optim as optim
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
		'''compute output of (batch size) * Si * Sj * 3
		'''
		for i,conv in enumerate(self.convs):
			if i in self.pools:
				x = self.pool(x)
			x = F.relu(conv(x))
		return x

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, steps: int):
	'''given an initialized yolo object, train
	'''
	rloss = 0

	lossf = nn.MSELoss()
	opt = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9)
	for i, (imgs, lbls) in enumerate(data):
		if i >= steps:
			break

		# optimization step: compute loss and
		# backpropagate
		opt.zero_grad()
		pred = yolo(imgs)
		loss = lossf(pred, lbls)
		loss.backward()
		opt.step()

		# expectation over loss, with recent loss more
		# important
		rloss = .9 * rloss + .1 * loss.item()
		print(f'running loss: {rloss}')
