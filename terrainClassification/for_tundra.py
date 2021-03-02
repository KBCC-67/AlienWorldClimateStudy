# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:29:06 2021

@author: Templar_KBCC
"""
'''LBP尝试，貌似这个库有问题用不了，我也不想自己编，应该先试简单的办法
import cv2
from skimage.feature import local_binary_pattern
from skimage import filters
import skimage
path="terrain\\\\earth.png"
# settings for LBP
radius = 1	# LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
image=cv2.imread(path)
image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lbp = local_binary_pattern(image_g, n_points, radius)
edges = filters.sobel(image_g)
cv2.imshow("a",edges)
cv2.destroyAllWindows()'''
#hsv色域+边缘多少程度二元k-means分类，只有两个用k-means非常直观
import cv2
import numpy as np
import matplotlib.pyplot as plt
path="terrain\\\\tundra.png"
img_origin=cv2.imread(path)
print(img_origin.shape)
img_copy=img_origin
img_blur=cv2.blur(img_origin,(7,7))
#1:hsv
HSV_img = cv2.cvtColor(img_origin,cv2.COLOR_BGR2HSV)

#2:边缘化特征
#提取边缘化特征之前先对亮的地方变暗处理，极地太亮了会有很多误判的山脉丘陵，或者模糊处理？
for i in range(img_copy.shape[0]):
    for j in range(img_copy.shape[1]):
        if HSV_img[i][j][2]>=80 and HSV_img[i][j][1]<=42:#s小说明像灰色
            #img_copy[i][j]=[int(img_copy[i][j][0]/1.5),int(img_copy[i][j][1]/1.5),int(img_copy[i][j][2]/1.5)]
            img_copy[i][j]=img_blur[i][j]
cv2.imwrite("darken.jpg",img_copy)
img=cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
cv2.imwrite("darken_edge.jpg",img)

kernel_h = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
result_h=cv2.filter2D(img,-1,kernel_h)
ret,bi_result_h = cv2.threshold(result_h,30,255,cv2.THRESH_BINARY)#注意前面要ret
kernel_v = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
result_v=cv2.filter2D(img,-1,kernel_v)
ret,bi_result_v = cv2.threshold(result_v,30,255,cv2.THRESH_BINARY)#注意前面要ret
#5x5作为区间判别潜在山地、丘陵、平原
result_h_v=cv2.add(result_h,result_v)/255#opencv的叠加超限时取上限而numpy是取模
mountain_ker=np.ones((5,5))
mountain_map=cv2.filter2D(result_h_v,-1,mountain_ker)
#ret,mountain=cv2.threshold(mountain_map,4,255,cv2.THRESH_BINARY)
#ret,hill=cv2.threshold(mountain_map,2,255,cv2.THRESH_BINARY)#hill包含了mountain
#点数统计
'''
x_mountain_degree=mountain_map.flatten()#二维矩阵拉成一维
print(x_mountain_degree.size)
y_h_degree=HSV_img[:,:,0:1]
y_h_degree=y_h_degree.flatten()
print(y_h_degree.size)
plt.scatter(x_mountain_degree, y_h_degree, s=1, marker=".")
plt.show()'''

#叠加图片可视化,我直接决策树
#现在是平原和极地的交界也当成山地
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if HSV_img[i][j][2]>=130 and HSV_img[i][j][1]<42:#白色地貌对地球取明度>=200,对联合取HSV_img[i][j][2]>=130 and HSV_img[i][j][1]<42
            if mountain_map[i][j]>=10:
                img_origin[i][j]=[255,0,153]#紫色，山地
            elif mountain_map[i][j]>=5:
                img_origin[i][j]=[37,65,204]#棕色，丘陵
            else:
                img_origin[i][j]=[255,255,255]#白色，雪原
        elif HSV_img[i][j][0]>=19 and HSV_img[i][j][0]<=77:#绿色、青色地貌
            if mountain_map[i][j]>=4:
                img_origin[i][j]=[255,0,153]#紫色，山地
            elif mountain_map[i][j]>=2:
                img_origin[i][j]=[37,65,204]#棕色，丘陵
            else:
                img_origin[i][j]=[0,255,0]#绿色，平原或森林
        elif HSV_img[i][j][0]>=78 and HSV_img[i][j][0]<=124:#深蓝色地貌
            if mountain_map[i][j]<6 and HSV_img[i][j][2]>100:
                img_origin[i][j]=[255,255,255]#白色，雪原
            elif mountain_map[i][j]<10 and HSV_img[i][j][2]>100:
                img_origin[i][j]=[37,65,204]#棕色，丘陵
            elif mountain_map[i][j]>=10 and HSV_img[i][j][2]>100:
                img_origin[i][j]=[255,0,153]#紫色，山地
            else:
                img_origin[i][j]=[255,0,0]#蓝色，水域
        else:#黄色、橙色、红色、浅蓝色地貌
            if mountain_map[i][j]>=7:
                img_origin[i][j]=[255,0,153]#紫色，山地
            elif mountain_map[i][j]>=4:
                img_origin[i][j]=[37,65,204]#棕色，丘陵
            else:
                img_origin[i][j]=[102,217,255]#黄色，沙漠
            
cv2.imwrite("horizontal.jpg",bi_result_h)
cv2.imwrite("verticl.jpg",bi_result_v)
cv2.imwrite("result_tundra.jpg",img_origin)
cv2.destroyAllWindows()