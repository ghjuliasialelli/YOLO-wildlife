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
				outsize = 3 if i+j == ndepth+nlayers-2 else 64
				print(f'in: {insize}, out: {outsize}')
				self.convs.append(nn.Conv2d(
					insize, outsize, 3, padding=1+x,dilation=1+x))
				x = 3*x+2
				setattr(self, f'conv{i}_{j}', self.convs[-1])
			if i != ndepth - 1:
				self.convs.append(nn.Conv2d(64,64,3,stride=3))  # pooling2.0
				setattr(self, f'conv{i}_{j}_', self.convs[-1])
				print(f'in: 64, out: 64')
	
	def forward(self,x):
		'''compute output of (batch size) * Si * Sj * 3
		'''
		for i,conv in enumerate(self.convs):
			print(x.size())
			# always use relu except last layer use sigmoid
			f = F.sigmoid if i == len(self.convs)-1 else F.relu
			x = f(conv(x))
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

		print(imgs.shape, lbls.shape)
		import numpy as np
		imgs = torch.tensor(imgs.transpose(0, 3, 1, 2).astype(np.float32)).type('torch.FloatTensor')
		lbls = torch.tensor(lbls.transpose(0, 3, 1, 2).astype(np.float32)).type('torch.FloatTensor')

		# optimization step: compute loss and
		# backpropagate
		opt.zero_grad()
		pred = yolo(imgs)
		loss = torch.mean(lossf(pred[:, :2, :, :], lbls[:, :2, :, :]) * lbls[:, 2, :, :]) + torch.mean(lossf(pred[:, 2, :, :], lbls[:, 2, :, :]))
		loss.backward()
		opt.step()

		# expectation over loss, with recent loss more
		# important
		rloss = .9 * rloss + .1 * loss.item()
		print(f'running loss: {rloss}')


if __name__ == '__main__':
	yolo = YOLO()
	train(yolo, datagen('downscaled1000x750', 'labels1000x750.json', 83, 111, 10), 1e-3, 100)
