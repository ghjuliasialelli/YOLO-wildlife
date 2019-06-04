import copy

Bbox = namedtuple('Bbox', 'x1 y1 x2 y2')

Cbox = namedtuple('Cbox', 'x y w h')


def IOU(B5, labels):
	for i in range():

		lt = torch.max(
            box1[:, :2],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, 0] * wh[:, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

        iou = inter / (area1 + area2 - inter)
        return iou

def center_to_bounding(box):
	for i in range(0,len(box),5):
		x,y,w,h = box[i:i+4]
		x1,y1,x2,y2 = (x-w,y-h,x+w,y+h)



#pre_labels are namedtuples = (sx,sy,x,y,h,w)
Label = namedtuple('Label', 'sx sy x y h w')
def get_label(Y, labels):
	Y = copy.deepcopy(Y)
	n = len(Y)
	for i in range(n):
		for l in labels[i]:
			for Y[i][l.sy][l.sx]:




def grid_labels(
