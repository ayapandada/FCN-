import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import caffe

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('coast_arnat59.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy32.prototxt', 'fcn32s_train_iter_40000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score_sem'].data[0].argmax(axis=0)

# visualize segmentation in PASCAL VOC colors)
plt.imshow(out)
plt.axis('off')
plt.savefig('coast_arnat59_fcn32s.jpg')
