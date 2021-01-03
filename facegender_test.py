import torchvision.models as models
import torch.nn as nn
import torch
import torch.utils.data as Data
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class mydataset(Data.Dataset):
    def __init__(self, path):
        self.image = []
        self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.path = path

    def __getitem__(self, item):
        image = Image.open('/'.join((self.path, '%d'%(item+1), 'face.jpg')))
        face = self.transform(image)
        return face

    def __len__(self):
        return len(os.listdir(self.path))

BATCH_SIZE=50
test_data = mydataset("/home/zyc/PycharmProjects/s2f/TRAIN")
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

model=models.resnet34(pretrained=False)
model.fc=nn.Linear(512, 2)
model.load_state_dict(torch.load("/home/zyc/PycharmProjects/f2s-pytorch/facegender/Classifier_epoch10.pkl"))
model = model.cuda()
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

for step, (x) in enumerate(test_loader):
    face = x.cuda()
    output = model(face)
    label = torch.max(output, dim=1)[1].data.cpu().numpy()
    with open("/home/zyc/PycharmProjects/s2f/train_face_gender.txt", 'a') as txt:
        for gender in label:
            content = "{}".format(gender)
            txt.write(content + '\n')
    print(step+1)