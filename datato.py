# import os
# import random
# import shutil
# #path1 = "/home/zyc/PycharmProjects/Hearing_Face/image_test"
# #path2 = "/home/zyc/PycharmProjects/Hearing_Face/specgram_test_heat"
# path1 = "/home/tione/notebook/VGG_ALL_FRONTAL"
# namefile = os.listdir(path1)
# L_name = len(namefile)
# def tripe(name):
#     file = open("/home/tione/notebook/untitled9_face_decoder/val.txt", 'a')
#     file.write('\n'+ name + ' ')
#     file.close()

# item = 0
# for name in namefile:
#     print(name)
#     path2 = path1 + '/' + name
#     for name1 in os.listdir(path2):
#         filename2 = path2 + '/' + name1
#         print(filename2)
#         tripe(filename2)
# #         item += 1
# #     print(item, name)


import os
import random
import shutil
#path1 = "/home/zyc/PycharmProjects/Hearing_Face/image_test"
#path2 = "/home/zyc/PycharmProjects/Hearing_Face/specgram_test_heat"
path1 = "E:/VGG_ALL_FRONTAL/VGG_ALL_FRONTAL"
namefile = os.listdir(path1)
L_name = len(namefile)
def tripe(name,item):
    imgpath = '/'.join((path1,name))

    imglist = os.listdir(imgpath)
    L = len(imglist)
    i = random.randint(0,L-1)
    image = imglist[i]
    image_path = '/'.join((imgpath,image))
    triple = ''.join((image_path,' '))
    file = open("E:/PycharmProjects/facedecoder/test.txt", 'a')
    file.write('\n'+triple)
    file.close()

item = 0
for name in namefile:
    for _ in range(1):
        tripe(name, item)
        item += 1
    print(item, name)