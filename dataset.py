import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd


class Data_ganerator(Dataset):
    def __init__(self,txt_file):
        with open(txt_file, 'r') as f:
            self.all_triplets = f.readlines()
        self.all_triplets = self.all_triplets[0:]
        self.transform1 = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        triplet = self.all_triplets[item].split(' ')
        image = triplet[0]
        face = self.load_image(image)
        return face


    def __len__(self):

        return len(self.all_triplets)

    def load_image(self,image_dir):
        image = Image.open(image_dir).convert('RGB')
        if image.size != (224, 224):
            image = image.resize((224, 224), resample=Image.BILINEAR)
        image = self.transform1(image)
        print('1111111111111111111',image)
        return image



if __name__ == '__main__':
    dataset = Data_ganerator('/home/tione/notebook/untitled9_face_decoder/val.txt')
    loader = DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=2)
    for step, (face) in enumerate(loader):
        print(face.shape)
       





