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
def test():
    # dataloader = Dataset('/home/yxy/s2f/data','/home/yxy/s2f/biplects','test',False)
    test_data = Data_ganerator('E:/PycharmProjects/facedecoder/test.txt')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net1 = VGG_Model()
    net2 = Face_Decoder()
    
    print('Load pretrained model')
    net1 = VGG_Model().to(device)
    checkpoint = torch.load('E:/PycharmProjects/facedecoder/vgg_face_dag.pth',map_location='cpu')
    net1.load_state_dict(checkpoint)
    for para in net1.parameters():
        para.requires_grad = False
    net1.eval()

    net2 = Face_Decoder().to(device)
    checkpoint2 = torch.load('E:/PycharmProjects/facedecoder/weight_epoch_68.pth',map_location='cpu')
    net2.load_state_dict(checkpoint2)
    for para in net2.parameters():
        para.requires_grad = False
    net2.eval()

    L1 = nn.PairwiseDistance(1)
    L2 = nn.PairwiseDistance(2)
    print('start test...')
    with torch.no_grad():
        for step, (face) in enumerate(test_loader):
            # print(face.shape)
            face = face.to(device)
            feat_7, _ = net1(face)
            
            output2 = net2(feat_7)
            #output2 = (output2 + 1
            print(output2.shape)
            print('22222222',output2)

            image = output2.cpu().clone() 
            print(image.shape)
            image = image.squeeze(0)  
            image = unloader(image)
            image.save('E:/PycharmProjects/facedecoder/demo/%d.jpg'%(step+1))
if __name__ == '__main__':
    test()