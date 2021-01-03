import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data1_loss =np.loadtxt("E:/PycharmProjects/Code_of_Speech2Face/TRAIN_loss_log2019-12-10-12-03-40.txt")
data2_loss = np.loadtxt("E:/PycharmProjects/Code_of_Speech2Face/TRAIN_loss_log2019-12-29-01-56-59.txt")
print(data1_loss)
#x = data1_loss[:,0]
x = np.arange(23381)
y = data1_loss[:,3]
#x1 = data2_loss[:,0]
x1 = np.arange(23381)
y1 = data2_loss[:,3]
print('111111',data2_loss[:,0])
print(x1)
print(y1)
fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`


# plt.gcf().set_facecolor(np.ones(4)* 240 / 255)   # 生成画布的大小
plt.grid()  # 生成网格
#y = np.linspace(-3,3)  #设置横轴的取值点
pl.plot(x,y,'r-',label=u'nuetral_prior')
# # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
pl.plot(x1, y1,'b-', label = u'no_prior')


#plt.plot(x,y,color='red',marker='^',label=u'nuetral_prior',linestyle=":",markerfacecolor="black",linewidth=1,markersize=20)#color表示整个线的颜色
#plt.plot(x1, y1,color='blue',marker='d',label=u'no_prior',linestyle=":",markerfacecolor="red",linewidth=1,markersize=20)#color表示整个线的颜色

#显示图例
#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend(fontsize = 18)
pl.xlabel(u'iters',fontsize=20)
pl.ylabel(u'loss',fontsize=20)

plt.title('Compare loss for different models in training',fontsize=18)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pylab import *
# data1_loss =np.loadtxt("E:/PycharmProjects/Code_of_Speech2Face/TRAIN_loss_log2019-12-10-12-03-40.txt")
# data2_loss = np.loadtxt("E:/PycharmProjects/Code_of_Speech2Face/TRAIN_loss_log2019-12-29-01-56-59.txt")
# print(data1_loss)
# #x = data1_loss[:,0]
# x = np.arange(23381)
# y = data1_loss[:,3]
# #x1 = data2_loss[:,0]
# x1 = np.arange(23381)
# y1 = data2_loss[:,3]
# print('111111',data2_loss[:,0])
# print(x1)
# print(y1)
#
# FontSize = 15
# x = range(0, 23381)
# # 下面四句对应图中四条线的纵坐标，因为x的维度为1*24,所以这四个变量的维度也是1*24,用的时候输入自己的坐标即可
#
# ax = plt.subplot(111)
#
# MarkerSize = 6  # 统一设置下面四句中的markersize，表示图中正方形、圆形、三角形、菱形等形状的大小
# plt.plot(x, y, color='black', marker='^', label=u'OurMethod', linestyle=":", markerfacecolor="black", linewidth=1,
#          markersize=MarkerSize)  # color表示整个线的颜色
# plt.plot(x1, y1, color='red', marker='d', label=u'EM-GCWC', linestyle=":", markerfacecolor="red", linewidth=1,
#          markersize=MarkerSize)  # color表示整个线的颜色
# plt.legend(fontsize=10)  # fontsize设置左上角图例中形状的大小
# plt.xticks(fontsize=12)  # fontsize设置x轴和y轴标度的大小
# plt.yticks(fontsize=12)
# plt.xlabel("Hours of Day", fontsize=FontSize)  # fontsize设置X轴标签的大小
# plt.ylabel("Percentage(%)", fontsize=FontSize)  # fontsize设置Y轴标签的大小
# plt.grid()  # 在背景中加上方格
# #plt.savefig('自己的目录/picture1.pdf', bbox_inches='tight')
# plt.show()
