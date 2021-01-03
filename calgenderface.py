import torch
import torch.nn as nn
import numpy as np
from face_decoder import VGG_Model
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import os
from PIL import Image

BATCH_SIZE = 1



class mydataset(Data.Dataset):
    def __init__(self, path, gender):
        if gender not in ['m','f']:
            print("gender is wrong! gender must be 'm' or 'f'")
        g = '1' if gender == 'm' else '0'
        self.faces = []
        with open("/home/zyc/PycharmProjects/s2f/train_face_gender.txt") as txt:
            content = txt.read().splitlines()
            i = 1
            for line in content:
                if line == g:
                    self.faces.append(i)
                i += 1
        self.path = path
        #self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        # for id in os.listdir(path):
        #     if infor[id] != gender:
        #         continue
        #     for face in os.listdir(path+'/'+id)[:10]:
        #         self.faces.append('/'.join((path, id, face)))

    def __getitem__(self, item):
        img = Image.open('/'.join((self.path,'%d'%self.faces[item],'face.jpg')))
        face = self.transform(img)
        img.close()
        return face

    def __len__(self):
        return 10000  #len(self.faces)

encoder = VGG_Model()
encoder.load_weights()
encoder = encoder.cuda()
encoder.eval()
for parameter in encoder.parameters():
    parameter.requires_grad = False
train_data = mydataset("/home/zyc/PycharmProjects/s2f/TRAIN",'m')
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
length = len(train_data)
mean_maleface = torch.zeros((4096))
mean_maleface = mean_maleface.cuda()
for step, (x) in enumerate(train_loader):
    faces = x.cuda()
    output, _ = encoder(faces)
    sumface = torch.sum(output, dim=0)
    mean_maleface += sumface
    print(step+1)
mean_maleface = mean_maleface / length
print(mean_maleface)

train_data = mydataset("/home/PycharmProjects/s2f/TRAIN",'f')
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
length = len(train_data)
mean_femaleface = torch.zeros((4096))
mean_femaleface = mean_femaleface.cuda()
for step, (x) in enumerate(train_loader):
    faces = x.cuda()
    output, _ = encoder(faces)
    sumface = torch.sum(output, dim=0)
    mean_femaleface += sumface
    print(step+1)
mean_femaleface = mean_femaleface / length
mean_face = torch.cat((mean_maleface.unsqueeze(dim=0),mean_femaleface.unsqueeze(dim=0)),dim=0)
mean = mean_face.data.cpu().numpy()
# np.save("/home/zyc/PycharmProjects/s2f/meanface.npy", mean)
np.save("/Samsung_T5/priorface/10000.npy", mean)

from face_decoder import Face_Decoder
decoder = Face_Decoder()
checkpoint = torch.load('F:/facedecoder/weight_epoch_99.pth')
state_dict = checkpoint['net']
decoder.load_state_dict(state_dict)
decoder = decoder.cuda()
for parameter in decoder.parameters():
    parameter.requires_grad = False
unloader = transforms.ToPILImage()
face_output = decoder(mean_maleface)
image = face_output.cpu().clone()
image = image.squeeze(0)
image = unloader(image)
image.save('/Samsung_T5/priorface/10000male.jpg')

face_output = decoder(mean_femaleface)
image2 = face_output.cpu().clone()
image2 = image2.squeeze(0)
image2 = unloader(image2)
image2.save('/Samsung_T5/priorface/10000female.jpg')