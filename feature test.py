import torch.nn as nn
from face_decoder import VGG_Model
import os
import numpy as np
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F




def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

class s2f_Dataset(data.Dataset):
    # def __init__(self, path1, path2)://all
    #     self.datapath1 = path1
    #     self.datapath2 = path2
    #     self.length = len(os.listdir(path1))
    #     self.transform = transforms.Compose(
    #         [transforms.ToTensor()]
    #     )
    def __init__(self, path1, path2):
        # if gender not in ['m','f']:
        #     print("gender is wrong! gender must be 'm' or 'f'")
        # g = '1' if gender == 'm' else '0'

        self.datapath1 = path1
        self.datapath2 = path2
        #self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, item):
        img1 = Image.open('/'.join((self.datapath1, '%d'%(item+1)+'.jpg')))
        face1 = self.transform(img1)
        img1.close()
        img2 = Image.open('/'.join((self.datapath2, 's2f'+'%d'%(item+1)+'.jpg')))
        face2 = self.transform(img2)
        img2.close()
        input = [face1, face2]
        return input
        # img1 = Image.open('/'.join((self.path1, '%d' % self.faces[item]+'.jpg')))
        # face1 = self.transform(img1)
        # img1.close()
        # img2 = Image.open('/'.join((self.path2, 's2f'+'%d' % self.faces[item]+'.jpg')))
        # print(img2)
        # face2 = self.transform(img2)
        # img2.close()
        # print('/'.join((self.path1, '%d' % self.faces[item]+'.jpg')))
        # print('/'.join((self.path2, 's2f'+'%d' % self.faces[item]+'.jpg')))
        # input = [face1, face2]
        # return input

    def __len__(self):
        #return self.length //all
        return 5000

if __name__ == '__main__':
    test_data = s2f_Dataset("k:/TEST22","C:/Users/hxs/Desktop/result/s2f")
    len_data = len(test_data)
    print('len_data',len_data)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=1)

    print("Loading model")
    net1 = VGG_Model()
    #net1 = net1.cuda()
    #net1.load_weights()
    checkpoint = torch.load('K:/facedecoder/vgg_face_dag.pth',map_location='cpu')
        #net1.load_weights()
    net1.load_state_dict(checkpoint)
    for para in net1.parameters():
        para.requires_grad = False
    net1.eval()


    print("Done!")
    print("Starting train")

    L1 = nn.PairwiseDistance(1)
    L2 = nn.PairwiseDistance(2)
    tL1, tL2, tcos = 0, 0, 0
    for step, (x) in enumerate(test_loader):

        # face1 = x[0].cuda()
        # face2 = x[1].cuda()
        face1 = x[0]
        face2 = x[1]
        feat1, _ = net1(face1)
        feat2, _ = net1(face2)
        print(L1(feat1, feat2))
        print(L2(feat1, feat2))
        tL1 += L1(feat1, feat2).item()
        tL2 += L2(feat1, feat2).item()
        tcos += cos_sim(feat1.data.cpu().numpy(), feat2.data.cpu().numpy())
        print(step)
    tL1 /= len_data
    tL2 /= len_data
    tcos /= len_data
    print(tL1, tL2, tcos)






