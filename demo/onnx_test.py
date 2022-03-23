import sys
import onnx
import onnxruntime as rt
import numpy as np
from PIL import Image
import cv2
import time

import torch

from mmdet.core.bbox.coder import DeltaXYWHBBoxCoder
from mmrotate.models.detectors import RotatedRetinaNet

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

img = cv2.imread('linhdam_005042.png')
in_frame = cv2.resize(img, (1024, 1024))
X = np.asarray(in_frame)
X = X.astype(np.float32)
X = X.transpose(2,0,1)
# Reshaping the input array to align with the input shape of the model
X = X.reshape(1,3,1024,1024)




sess = rt.InferenceSession("tmp.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
# scores = sess.run([output_name], {input_name: np.random.random((1,3,1024,1024)).astype(np.float32)})[0]
# print (scores[0])

start = time.time()
raw_result = sess.run([], {input_name: X})
end = time.time()


print("Output: ")
coder = DeltaXYWHBBoxCoder()

# cls = raw_result[:5]
# bbox = raw_result[:-5]

# print()

# print(len(raw_result))
# for out in raw_result:
#     print(out.shape)

# bbox_result = raw_result[0]

# bboxes = np.vstack(bbox_result)
# labels = [
#     np.full(bbox.shape[0], i, dtype=np.int32)
#     for i, bbox in enumerate(bbox_result)
# ]
# labels = np.concatenate(labels)





