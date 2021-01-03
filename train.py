import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import loss_ms
from dataset import Data_ganerator

from face_decoder import Face_Decoder
from face_decoder import VGG_Model
import torch.nn.functional as F
# import warping_inv_pyt as warp
import numpy as np


def train():
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # dataloader = Dataset('/home/yxy/s2f/data','/home/yxy/s2f/biplects','test',False)

    train_data = Data_ganerator('/home/tione/notebook/untitled9_face_decoder/train.txt')
    train_loader = DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True,num_workers=2)
    val_data = Data_ganerator('/home/tione/notebook/untitled9_face_decoder/val.txt')
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, drop_last=True, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    net1 = VGG_Model().to(device)
    checkpoint = torch.load('/home/tione/notebook/untitled9_face_decoder/vgg_face_dag.pth')
    net1.load_state_dict(checkpoint)
    for para in net1.parameters():
        para.requires_grad = False
    net1.eval()

    net2 = Face_Decoder()
    net2.apply(weight_init)
    print('Net2 Initialize weight with xavier_uniform.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net2 = Face_Decoder().to(device)


    loss_function_L2 = nn.MSELoss()
    loss_function_L1 = nn.L1Loss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    loss_function = loss_ms.MS_SSIM(max_val=1)
   
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.01)
    torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.01)
    
    weigth_save_dir = '/home/tione/notebook/untitled9_face_decoder/weight'
    
    step_size = int(len(train_data)/64) #batch_size
    val_step_size = int(len(val_data)/64) #batch_size
    print('start trainning...')
    for epoch in range(100):
        net2.train()
        for step, (face) in enumerate(train_loader):

            face = face.to(device)
            feat_7, _ = net1(face)
            output2 = net2(feat_7)
            feat_7_1 = net1(output2)[0]

            loss1 = (1 - loss_function(face, output2)) * 0.84 + 0.16 * loss_function_L1(face, output2)
            loss2 = torch.mean(1 - cos(feat_7, feat_7_1), dim=0)
            Total_Loss = loss1 + loss2
            optimizer2.zero_grad()
            Total_Loss.backward()

            optimizer2.step()
            print('Epoch[{}/{}]  Step[{}/{}]  Loss1: {:.8f} Loss2: {:.8f}  Total_loss: {:.8f}  '.format(
                epoch + 1, 100, step + 1, step_size, loss1.item(), loss2.item(),Total_Loss.item()))
            # VALIDATING
        
        save_weight_file = 'weight_epoch_{}.pth'.format(epoch)
        torch.save(net2.state_dict(), os.path.join(weigth_save_dir, save_weight_file))
        
        print('start val...')
        totalvalloss = 0
        nval = 0
        net2.eval()
        for step, (face) in enumerate(val_loader):
            face = face.to(device)
            feat_7, _ = net1(face)
            output2 = net2(feat_7)
            feat_7_1 = net1(output2)[0]
            loss1 = (1 - loss_function(face, output2)) * 0.84 + 0.16 * loss_function_L1(face, output2)
            loss2 = torch.mean(1 - cos(feat_7, feat_7_1), dim=0)
            totalvalloss = loss1 + loss2
            nval += 1
            print('Epoch[{}/{}]  Step[{}/{}]  Loss1: {:.8f} Loss2: {:.8f}  '.format(
                epoch + 1, 100, step + 1, val_step_size, loss1.item(), loss2.item()))
        totalvalloss = totalvalloss / nval
        with open("/home/tione/notebook/untitled9_face_decoder/VAL_loss_log.txt", 'a') as txt:
            content = "{}".format(totalvalloss)
            txt.write(content + '\n')
        # SAVE THE CHECKPOINT


if __name__ == '__main__':
    train()

