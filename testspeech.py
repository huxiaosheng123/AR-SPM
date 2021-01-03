import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset1 import s2f_Dataset
import importlib
from face_decoder import Face_Decoder
from face_decoder import VGG_Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
from speechencoder import SpectrogramNet

if __name__=='__main__':
    unloader = transforms.ToPILImage()
    BATCH_SIZE = 1
    test_data = s2f_Dataset("F:/TEST2")
    #test_data = s2f_Dataset('C:/Users/hxs/Desktop/test')
    #test_data = s2fgender_Dataset("D:/BaiduNetdiskDownload/TEST2",'m')
    len_data = len(test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    #neutral = torch.from_numpy(np.load("E:/PycharmProjects/Code_of_Speech2Face/normal_meanface.npy"))
    neutral = torch.from_numpy(np.load("F:/facedecoder/normal_meanface.npy"))
    #print(neutral)
    # male, female, neutral = torch.from_numpy(np.load("/home/zyc/PycharmProjects/s2f/cn_meanface.npy"))
    print("Loading model")

    # decoder = Face_Decoder()
    # checkpoint = torch.load('E:/PycharmProjects/Code_of_Speech2Face/model_50_1208.pt',map_location = 'cpu')
    # state_dict = checkpoint['net']
    # decoder.load_state_dict(state_dict)
    # decoder = decoder.cuda()

    decoder = Face_Decoder()
    checkpoint2 = torch.load('F:/facedecoder/weight_epoch_99.pth',map_location='cpu')#43
    decoder.load_state_dict(checkpoint2)

    mapping = SpectrogramNet()
    mapping.load_state_dict(torch.load('F:/facedecoder/neutralprior.pkl', map_location='cpu'))
    #mapping.load_state_dict(torch.load('I:/1/EPOCH70_mapping.pkl', map_location='cpu'))


    net1 = VGG_Model()
    #net1 = net1.cuda()
    checkpoint = torch.load('F:/facedecoder/vgg_face_dag.pth',map_location='cpu')
        #net1.load_weights()
    net1.load_state_dict(checkpoint)
    # net1.load_weights()
    for para in net1.parameters():
        para.requires_grad = False


    for parameter in decoder.parameters():
        parameter.requires_grad = False

    for parameter in mapping.parameters():
        parameter.requires_grad = False

    print("Done!")
    print("Starting train")
    net1.eval()
    decoder.eval()
    mapping.eval()

    L1 = nn.PairwiseDistance(1)
    L2 = nn.PairwiseDistance(2)
    tl1, tl2, tcos = 0, 0, 0
    def cos_sim(vector_a, vector_b):
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


    list1 = []
    for step, (x) in enumerate(test_loader):
        # audio = x[1].cuda()
        # face = x[0].cuda()
        audio = x[1]
        print(audio.shape)
        face = x[0]

        minib = face.size(0)
        mean = torch.zeros((minib, 4096))
        for i in range(minib):
            mean[i, :] = neutral
       # mean = mean.cuda()
        print()
        face_feat7, _ = net1(face)
        s2f_feat = mapping(audio, fuse=True, mean=mean)
        output = decoder(face_feat7)
        face_output = decoder(s2f_feat)
        # if a_min>a :
        #     a_min = a
        # if b_min>b :
        #     b_min =b
        # if c_min<c:
        print(L1(face_feat7,s2f_feat))
        print(L2(face_feat7, s2f_feat))
        if L1(face_feat7,s2f_feat) >=30:
            with open("k:/facedecoder/remove.txt", 'a') as txt:
                content = "{}".format(step+1)
                txt.write(content + '\n')
            # SAVE THE CHECKPOINT
        image = output.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        #image.save('I:/facedecoder/result/f2f/f2f%d.jpg'%(step+1))
        image.save('C:/Users/hxs/Desktop/result/f2f/f2f%d.jpg'%(step+1))

        image2 = face_output.cpu().clone()
        image2 = image2.squeeze(0)
        image2 = unloader(image2)
        image2.save('C:/Users/hxs/Desktop/result/s2f/s2f%d.jpg' % (step + 1))

        print(step)
        tl1 += (L1(s2f_feat, face_feat7)).item()
        tl2 += (L2(s2f_feat, face_feat7)).item()
        tcos += cos_sim(s2f_feat.data.cpu().numpy(), face_feat7.data.cpu().numpy())

    print(tl1 / 5000)
    print(tl2 / 5000)
    print(tcos / 5000)
    print(list1)