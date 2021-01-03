import torch.nn as nn
import torch
import numpy as np
def convert(a, T):
    a_exp = torch.exp(a / T)
    P_a = a_exp / torch.unsqueeze(torch.sum(a_exp, dim=1),dim=1)
    return P_a

def distill(a, b, T):
    cos = nn.CosineSimilarity(dim=1)
    distill_loss = torch.mean((1 - cos(b, a)), dim=0)

    # logP = nn.LogSoftmax(dim=1)
    #print("f:",a.data.cpu().numpy())
    #print("s:",b.data.cpu().numpy())
    # P_a = P(a/T)
    #P_b = logP(b/T)
    # print(torch.max(P(b),dim=1)[1])
    # print(P_a.data.cpu().numpy())
    # print(P(b).data.cpu().numpy())
    # distill_loss = torch.mean(torch.neg(torch.sum(torch.mul(P_a, P_b), dim=1)), dim=0)
    return distill_loss

def norm2(Vf, Vs):
    norm2_Vf = torch.unsqueeze(torch.norm(Vf, p=2, dim=1), dim=1)
    norm2_Vs = torch.unsqueeze(torch.norm(Vs, p=2, dim=1), dim=1)
    unit_Vf = Vf / norm2_Vf
    unit_Vs = Vs / norm2_Vs
    #loss_norm = torch.mean(torch.pow(torch.norm(unit_Vf-unit_Vs, p=2, dim=1), 2), dim=0)
    L2 = nn.MSELoss()
    loss_norm = 4096*L2(unit_Vs,unit_Vf)
    return loss_norm

def loss_total(Vs, Vs_dec, Vs_VGG, Vf, Vf_dec, Vf_VGG, lamda1=1, lamda2=1, lamda3=0.04, T=2.0, printloss=False, loss_log=None):
    L1 = nn.L1Loss()
    #L2 = nn.MSELoss()
    loss_dec = L1(Vs_dec, Vf_dec)
    loss_VGG = distill(Vf_VGG, Vs_VGG, T)
    #loss_VGG = L1(Vs_VGG, Vf_VGG)
    loss_norm = norm2(Vf, Vs)
    # loss_norm = L2(Vs, Vf)
    loss_total = (lamda1 * loss_norm) + (lamda2 * loss_VGG) + (lamda3 * loss_dec)
    if printloss == True:
        # print('Vs:', Vs.data.cpu().numpy())
        # print('Vf:', Vf.data.cpu().numpy())
        loss1 = loss_norm.item()
        print('loss of vsvf:',loss1)
        loss2 = loss_VGG.item()
        print('loss of encoder:', loss2)
        loss3 = loss_dec.item()
        print('loss of Decoder:', loss3)
        # loss4 = GD_fake_loss.item()
        # print('loss of discriminator:', loss4)
        losst = loss_total.item()
        print('Total loss :', losst)
        with open(loss_log,'a') as txt:
            content = "{} {} {} {}".format(loss1, loss2, loss3, losst)
            txt.write(content+'\n')
    return loss_total