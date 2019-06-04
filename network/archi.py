import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from typing import Iterable, Tuple
from augment import datagen, Imgs, Labels

"""
this is the shape of our model
the dotted horizontal lines are convolutional layers
which play the role of a pooling
the pyramids are the series of convolutions
where we increase the dilation such that each final 
node has seen each pixel exactly once

   /|\ 
  / | \ 
 /  |  \ 
 -------
   /|\ 
  / | \ 
 /  |  \ 
 -------
   /|\ 
  / | \ 
 /  |  \ 

"""
nlayers = 3

ndepth = 3

class YOLO(nn.Module):
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''        
		super(YOLO, self).__init__()
		self.convs = []
		for i in range(ndepth):
			x = 0
			for j in range(nlayers):
				insize = 3 if i+j == 0 else 64
				outsize = 3 if i+j == ndepth+nlayers else 64
				self.convs.append(nn.Conv2d(
					insize, outsize, 3, padding=1+x,dilation=1+x))
				x = 3*x+2
			if i != ndepth:
				self.convs.append(nn.Conv2d(64,64,3,stride=3))  # pooling2.0
	
	def forward(self,x):
		'''compute output of (batch size) * Si * Sj * 3
		'''
		for i,conv in enumerate(self.convs):
			# always use relu except last layer use sigmoid
			f = F.sigmoid if i == len(self.convs) else F.relu
			x = f(conv(x)
		return x

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, steps: int):
	'''given an initialized yolo object, train
	'''
	rloss = 0

	# we treat this as a regression
	lossf = nn.MSELoss()
	opt = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9)
	for i, (imgs, lbls) in enumerate(data):
		if i >= steps:
			break

		# optimization step: compute loss and
		# backpropagate
		opt.zero_grad()
		pred = yolo(imgs)
		loss = lossf(pred[..., :2], lbls[..., :2]) * lbls[.., 2] + lossf(pred[..., 2], lbls[..., 2])
		loss.backward()
		opt.step()

		# expectation over loss, with recent loss more
		# important
		rloss = .9 * rloss + .1 * loss.item()
		print(f'running loss: {rloss}')
