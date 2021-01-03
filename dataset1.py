import torch
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class s2f_Dataset(data.Dataset):
    def __init__(self, path):
        self.datapath = path
        #self.dir = os.listdir(path)
        self.length = len(os.listdir(path))
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, item):
        #audio = np.load('/'.join((self.datapath, self.dir[item], 'speech6s.npy')))
        #img = Image.open('/'.join((self.datapath, self.dir[item], 'face.jpg')))
        audio = np.load('/'.join((self.datapath, '%d'%(item+1), 'speech6s.npy')))
        img = Image.open('/'.join((self.datapath, '%d'%(item+1), 'face.jpg')))
        if img.size != (224, 224):
            image = img.resize((224, 224), resample=Image.BILINEAR)
        face = self.transform(img)
        img.close()
        audio = torch.from_numpy(audio).type(torch.FloatTensor)
        #audio = audio.contiguous().view(2, -1, 257)
        input = [face, audio]
        return input

    def __len__(self):
        return self.length
