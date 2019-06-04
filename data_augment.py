from skimage import transform
import numpy as np

# label : (i1 j1 i2 j2)

"""
brief : perfoms a transformation of the images (including
translation, rotation, and flip). Yield infinitely. Mini
batch de N, transform toutes. 

Translation
Rotation
Flip with probability 1/2

input : 
    imgs : a dict() whose keys are the ImageIDs, and 
    whose values are the actual images, whose shape
    are (1000,750,3).
    labels : a dict() whose keys are the ImageIDs, and
    whose values are a list of (x,y,w,h). The len of the
    list corresponds to the number of animals in the image.

output : 

transform with random parameters



Module used : skimage.transform.AffineTransform(matrix=None, 
    scale=None, rotation=None, shear=None, translation=None)

    matrix : (3, 3) array, optional


matrix : (3, 3) array, optional
scale : (sx, sy) as array, list or tuple, optional
    Scale factors.
rotation : float, optional
    Rotation angle in counter-clockwise direction as radians.
shear : float, optional
    Shear angle in counter-clockwise direction as radians.
translation : (tx, ty) as array, list or tuple, optional
    Translation parameters.

"""



# label translation : i1 j1 i2 j2 


def rotate_label(label, rot_matrix):
    vect1 = np.asarray(label[:2])
    vect2 = np.asarray(label[2:])

    new1 = rot_matrix @ vect1 
    new2 = rot_matrix @ vect2

    new_label = (new1[0], new1[1], new2[0], new2[1])

    return new_label

# We rotate with random angle of rotation
# Rotate image by a certain angle around its center.
# Returns the rotated image along with the new labels 
def rotate(img_copy, labels):
    rotation = transform.SimilarityTransform(scale = 1, rotation =  np.random.vonmises(0.0,200)) # educated choice
    image_rotated = transform.warp(img_copy, rotation)
    rot_matrix = rotation.params 

    new_labels = []
    for label in labels : 
        new_labels.append(rotate_label(label, rot_matrix))

    return image_rotated, new_labels



def flip_label(label, axis):
    if axis == 0 : 
        i1 = 1 - label.i2
        i2 = 1 - label.i1

        j1 = label.j1
        j2 = label.j2
    else : 
        j1 = 1 - label.j2
        j2 = 1 - label.j1

        i1 = label.i1
        i2 = label.i2

    new_label = (i1, j1, i2, j2)
    return new_label

"""
No skimage method designed to flip. Use numpy.flip(m, axis=None)
m : input array
axis = 0 : flip an array vertically
axis = 1 : flip an array horizontally
"""
# We flip 
# We return the flipped image along with new labels 
def flip(img, labels):
    axis = np.random.choice([True, False], 1)
    new_img = np.flip(img, axis)

    new_labels = []
    for label in labels : 
        new_labels.append(flip_label(label, axis))

    return new_img, new_labels



""" test for flip : 
from skimage.io import imread
im = imread("d7d0b51e8b124c55a70963a28b55b8b2.JPG")
im2 = flip(im)
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()
plt.imshow(im2)
plt.show()
"""


""" test for rotate : 
from skimage.io import imread
im = imread("d7d0b51e8b124c55a70963a28b55b8b2.JPG")
im2 = rotate(im)
from skimage.io import imshow
imshow(im2)
import matplotlib
matplotlib.pyplot.show()
"""



def augment(imgs, all_labels):
    # parameter N : size of mini-batch
    N = 10

    ids = imgs.keys()

    # we randomly select N images from the pool of imgs
    for idx in np.random.randint(0, len(imgs),N):
        im_id = ids[idx]
        img = imgs[im_id].copy()
        labels = all_labels[im_id]

        # Rotate image
        rotated_img, rotated_labels = rotate(img, labels)

        # Flip image with probability 1/2
        flip = np.random.choice([True, False],1)
        if(flip): flipped_img, flipped_labels = flip(img, labels)
        else : flipped_img, flipped_labels = img, labels

        for label in rotated_labels : 
            yield rotated_img, label
        for label in flipped_labels :
            yield flipped_img, label
        