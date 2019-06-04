import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from typing import Iterable, Tuple


# entirely for collaboration purposes
Bbox = namedtuple('Bbox', 'x y w h')
Imgs =   NewType('Img', np.array)     # shape = (bs, w, h, 3)
Labels = NewType('Labels', np.array)  # shape = (bs, S, S, 3)

structure = [

		]

class YOLO(nn.Module):
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''        
		super(YOLO, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(3, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(3, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(3, 64, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

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
