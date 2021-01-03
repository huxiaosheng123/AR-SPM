import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

meta = open("/home/zyc/PycharmProjects/Hearing_Face/vox1_meta.csv")
line = meta.readline()
line = meta.readline()
infor = {}
while line:
    line_split = line.split()
    name = line_split[1]
    gender = line_split[2]
    infor[name] = gender
    line = meta.readline()
meta.close()

class mydataset(Data.Dataset):
    def __init__(self, path):
        self.image = []
        self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        for id in os.listdir(path):
            gender = 1 if infor[id] == 'm' else 0
            for spe in os.listdir(path+'/'+id):
                self.image.append('/'.join((path, id, spe)))
                self.labels.append(gender)

    def __getitem__(self, item):
        image = Image.open(self.image[item])
        face = self.transform(image)
        label = np.asarray(self.labels[item])
        label = torch.from_numpy(label).long()
        #label = torch.unsqueeze(label, dim=0)
        return face, label

    def __len__(self):
        return len(self.labels)

model=models.resnet34(pretrained=True)
model.fc=nn.Linear(512, 2)
#model.num_class=2
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
loss_fc=nn.CrossEntropyLoss()

BATCH_SIZE=16
train_data = mydataset("/home/zyc/PycharmProjects/Hearing_Face/image_train")
test_data = mydataset("/home/zyc/PycharmProjects/Hearing_Face/image_test")
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
len_data = len(train_data)
len_testdata = len(test_data)
EPOCH = 10

for epoch in range(EPOCH):
    # scheduler1.step()
    scheduler.step()
    # mySN.train()
    model.train()
    right, right2 = 0, 0
    for step, (x, y) in enumerate(train_loader):
        # print(x.size(),y.size())
        spectrogram = Variable(x).cuda()
        real_y = Variable(y).cuda()
        output = model(spectrogram)
        loss = loss_fc(output, real_y)
        # optimizer1.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        # optimizer1.step()
        optimizer.step()
        pred_y = torch.max(output, dim=1)[1].data.cpu().numpy()
        right += sum(pred_y == real_y.data.cpu().numpy())
        print('Epch:', epoch, '  |  ', 'step:', step, '  |  ', 'loss:', loss.item())
    print("train accuracy", right*1.0/len_data)
    # mySN.eval()
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        spectrogram = x.cuda()
        output = model(spectrogram)
        pred_y = torch.max(output, dim=1)[1].data.cpu().numpy()
        right2 += sum(pred_y == y.data.cpu().numpy())
    print("test accuracy", right2 * 1.0 / len_testdata)
    torch.save(model.state_dict(), "/home/zyc/PycharmProjects/f2s-pytorch/facegender/Classifier_epoch%d.pkl" % (epoch + 1))