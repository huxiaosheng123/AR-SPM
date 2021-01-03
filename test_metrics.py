import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset import Data_ganerator
import importlib
from face_decoder import Face_Decoder
from face_decoder import VGG_Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2

unloader = transforms.ToPILImage()
transform1 = transforms.Compose([transforms.ToTensor()])

def load_image( image_dir):
    image = Image.open(image_dir).convert('RGB')
    if image.size != (224, 224):
        image = image.resize((224, 224), resample=Image.BILINEAR)
    image = transform1(image)
    print('1111111111111111111', image)
    return image

def test():
    L1 = nn.PairwiseDistance(1)
    L2 = nn.PairwiseDistance(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Load pretrained model')
    net1 = VGG_Model().to(device)
    checkpoint = torch.load('E:/PycharmProjects/facedecoder/vgg_face_dag.pth', map_location='cpu')
    net1.load_state_dict(checkpoint)
    for para in net1.parameters():
        para.requires_grad = False
    net1.eval()

    image1 = load_image(r'E:\PycharmProjects\result\f2f\f2f3.jpg')
    image2 = load_image(r'D:\BaiduNetdiskDownload\s2fTrueFace\s2fTrueFace\3.jpg')
    image1.unsqueeze_(0)
    image2.unsqueeze_(0)
    print(image1.shape)
    feat_7, _ = net1(image1)
    feat_71, _ = net1(image2)
    print('111111111')
    print(L1(feat_7,feat_71))
    print(L2(feat_7, feat_71))

if __name__ == '__main__':
    test()