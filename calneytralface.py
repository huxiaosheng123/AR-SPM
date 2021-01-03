import numpy as np
import torch
import torchvision.transforms as transforms
# male, female = np.load("/home/zyc/PycharmProjects/s2f/meanface.npy")
def L1(x,y):
    return np.sum(np.abs(x-y))
male, female = np.load("/Samsung_T5/priorface/10000.npy")
# male2, female2 = np.load("/media/zyc/Samsung_T5/priorface/5000.npy")
mean = (male + female) / 2
# mean2 = (male2 + female2) / 2
# print(male-male2,L1(male,male2), L1(female,female2), L1(mean,mean2))
np.save("/home/PycharmProjects/s2f/normal_meanface.npy", mean)

from face_decoder import Face_Decoder
decoder = Face_Decoder()
checkpoint = torch.load('F:/facedecoder/weight_epoch_99.pth')
state_dict = checkpoint['net']
decoder.load_state_dict(state_dict)
decoder = decoder.cuda()
for parameter in decoder.parameters():
    parameter.requires_grad = False
unloader = transforms.ToPILImage()
mean = torch.from_numpy(mean)
mean = mean.cuda()
face_output = decoder(mean)
image2 = face_output.cpu().clone()
image2 = image2.squeeze(0)
image2 = unloader(image2)
# image2.save('/media/zyc/Samsung_T5/priorface/10000.jpg')