import torch
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

# class s2f_Dataset(data.Dataset):
#     def __init__(self, path):
#         self.datapath = path
#         #self.dir = os.listdir(path)
#         self.length = len(os.listdir(path))
#         self.transform = transforms.Compose(
#             [transforms.ToTensor()]
#         )
#
#     def __getitem__(self, item):
#         #audio = np.load('/'.join((self.datapath, self.dir[item], 'speech6s.npy')))
#         #img = Image.open('/'.join((self.datapath, self.dir[item], 'face.jpg')))
#         audio = np.load('/'.join((self.datapath, '%d'%(item+1), 'speech6s.npy')))
#         img = Image.open('/'.join((self.datapath, '%d'%(item+1), 'face.jpg')))
#         face = self.transform(img)
#         img.close()
#         audio = torch.from_numpy(audio).type(torch.FloatTensor)
#         #audio = audio.contiguous().view(2, -1, 257)
#         input = [face, audio]
#         return input
#
#     def __len__(self):
#         return self.length


class s2fgender_Dataset(data.Dataset):
    def __init__(self, path, gender):
        if gender not in ['m','f']:
            print("gender is wrong! gender must be 'm' or 'f'")
        g = '1' if gender == 'm' else '0'
        self.faces = []
        with open(r"E:\PycharmProjects\Code_of_Speech2Face\attribute_truefaceofcntrain.txt") as txt:
            content = txt.readlines()
            print('cccccccccccccc',content)

            for i,line in enumerate(content):
                line = line.strip('\n').split(',')
                print(line)
                if line[1] == 'm':
                    self.faces.append(i+1) #弥补差1
                i += 1
            print(self.faces)
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
        #print('88888888888888888888',item)
        audio = np.load('/'.join((self.path, '%d' % self.faces[item], 'speech6s.npy')))
        img = Image.open('/'.join((self.path,'%d'% self.faces[item],'face.jpg')))
        print('/'.join((self.path, '%d' % self.faces[item], 'speech6s.npy')))
        print('/'.join((self.path,'%d'% self.faces[item],'face.jpg')))
        face = self.transform(img)
        img.close()
        audio = torch.from_numpy(audio).type(torch.FloatTensor)
        input = [face, audio]
        return input

    def __len__(self):
        return 3028

if __name__ == '__main__':
    train_data = s2fgender_Dataset("D:/BaiduNetdiskDownload/TEST2",'m')
    len_data = len(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=32)
    for step, (x) in enumerate(train_loader):
        # start = time.time()
        audio = Variable(x[1])
        face = x[0]
        # print(audio.shape)
        # print(face.shape)