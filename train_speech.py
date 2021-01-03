import torch
import numpy as np
import time
import os
import torch.optim as optim
from face_decoder import VGG_Model
from face_decoder import Face_Decoder
from speechencoder import SpectrogramNet
from new_loss import loss_total
from dataset1 import s2f_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from discriminator import discriminator

EPOCH = 50
BATCH_SIZE = 16
# LR = 0.0001
LR = 0.001
train_data = s2f_Dataset("/home/zyc/PycharmProjects/s2f/TRAIN")
len_data = len(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
val_data = s2f_Dataset("/home/zyc/PycharmProjects/s2f/TEST4")
val_loader = DataLoader(dataset=val_data, batch_size=25, shuffle=False, num_workers=2)
neutral = torch.from_numpy(np.load("F:/facedecoder/normal_meanface.npy"))
# male, female, neutral = torch.from_numpy(np.load("/home/zyc/PycharmProjects/s2f/cn_meanface.npy"))
encoder = VGG_Model()
encoder.load_weights()
encoder = encoder.cuda()
encoder.eval()

decoder = Face_Decoder()
checkpoint = torch.load('F:/facedecoder/weight_epoch_99.pth')
state_dict = checkpoint['net']
decoder.load_state_dict(state_dict)
decoder = decoder.cuda()
decoder.eval()

mapping = SpectrogramNet()
# mapping.load_state_dict(torch.load('/home/zyc/PycharmProjects/s2f/output/2019-10-26-13-46-37/EPOCH50_mapping.pkl'))
#mapping = mapping.cuda()
# d_net = discriminator()
# d_net = d_net.cuda()

for parameter in encoder.parameters():
    parameter.requires_grad = False
for parameter in decoder.parameters():
    parameter.requires_grad = False


optimizer = optim.Adam(mapping.parameters(), lr=LR, betas=(0.5, 0.999), eps=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
# d_optimizer = optim.Adam(d_net.parameters(), lr=LR, betas=(0.5, 0.999), eps=1e-4)
# d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.8)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [12, 18], gamma=0.1)
loss_func = loss_total
T = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# BCE = nn.BCELoss()
# real_label = torch.full((BATCH_SIZE, 1), 1)
# fake_label = torch.full((BATCH_SIZE, 1), 0)
# real_label, fake_label = real_label.cuda(), fake_label.cuda()
for epoch in range(EPOCH):
    scheduler.step()
    mapping.train()
    # TRAINING
    for step, (x) in enumerate(train_loader):
        # start = time.time()
        audio = Variable(x[1]).cuda()
        face = x[0].cuda()
        # print(index)
        minib = face.size(0)
        mean = torch.zeros((minib, 4096))
        for i in range(minib):
            mean[i,:] = neutral
        mean = mean.cuda()
        s2f_feat = mapping(audio, fuse=False, mean=mean)
        face_feat7, face_feat8 = encoder(face)
        s_vgg = encoder(s2f_feat, onlyfc8=True)
        f_dec = decoder(face_feat7, onlyf1=True)
        s_dec = decoder(s2f_feat, onlyf1=True)
        if (step+1)%10 == 0:
            print('Epoch: %d | step: %d' % (epoch+1, step+1))
            loss = loss_func(s2f_feat, s_dec, s_vgg, face_feat7, f_dec, face_feat8, printloss=True, loss_log="/home/PycharmProjects/s2f/TRAIN_loss_log{}.txt".format(T))
        else:
            loss = loss_func(s2f_feat, s_dec, s_vgg, face_feat7, f_dec, face_feat8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # VALIDATING
    totalvalloss = 0
    nval = 0
    mapping.eval()
    for step, (x) in enumerate(val_loader):
        audio = x[1].cuda()
        face = x[0].cuda()

        mean = torch.zeros((25, 4096))
        for i in range(25):
            mean[i, :] = neutral
        mean = mean.cuda()
        # res = mapping(audio)
        # s2f_feat = res + mean
        s2f_feat = mapping(audio, fuse=True, mean=mean)
        face_feat7, face_feat8 = encoder(face)
        s_vgg = encoder(s2f_feat, onlyfc8=True)
        f_dec = decoder(face_feat7, onlyf1=True)
        s_dec = decoder(s2f_feat, onlyf1=True)
        valloss = loss_func(s2f_feat, s_dec, s_vgg, face_feat7, f_dec, face_feat8)
        totalvalloss += valloss.item()
        nval += 1
    totalvalloss = totalvalloss / nval
    with open("/home/PycharmProjects/s2f/VAL_loss_log{}.txt".format(T),'a') as txt:
        content = "{}".format(totalvalloss)
        txt.write(content+'\n')
    # SAVE THE CHECKPOINT
    if (epoch+1) % 2 == 0:
        if os.path.exists("/home/PycharmProjects/s2f/output/%s"%T) == False:
            os.makedirs("/home/PycharmProjects/s2f/output/%s"%T)
        torch.save(mapping.state_dict(), "/home/PycharmProjects/s2f/output/%s/EPOCH%d_mapping.pkl" % (T, (epoch + 1)))
