# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:39:28 2021

@author: wwp
"""


import numpy as np
import cv2
rain_statistic=np.load("result\\\\rain_statistic.npy")
humidity_statistic=np.load("result\\\\humidity_statistic.npy")
temperature_statistic=np.load("result\\\\temperature_statistic.npy")
is_water_matrix=np.load("initial\\\\is_water_matrix.npy")
print("年平均气温 最高：",temperature_statistic.max()," 最低:",temperature_statistic.min()," 全球平均:",temperature_statistic.mean())
print("年平均湿度 最高：",humidity_statistic.max()," 最低:",humidity_statistic.min()," 全球平均:",humidity_statistic.mean())
print("年平均降雨 最高：",rain_statistic.max()," 最低:",rain_statistic.min()," 全球平均:",rain_statistic.mean())

temperature_color=np.zeros((1024,2048,3))#h从100到0，即蓝到红
temperature_color[:,:,0]=(temperature_statistic-temperature_statistic.min())/(temperature_statistic.max()-temperature_statistic.min())*100*(-1)+100
temperature_color[:,:,1]=np.ones((1024,2048))*255
temperature_color[:,:,2]=np.ones((1024,2048))*255
temperature_color_2=np.array(temperature_color,dtype=np.uint8)#这样才行
temperature_color_2=cv2.blur(temperature_color_2,(3,3))
temperature_color_2=cv2.cvtColor(temperature_color_2,cv2.COLOR_HSV2BGR)
cv2.imwrite("visualize\\\\temperature_color.jpg",temperature_color_2)

rain_statistic=rain_statistic*(1-is_water_matrix)
rain_color=np.zeros((1024,2048,3))#h从0到100，即红到蓝
rain_color[:,:,0]=(rain_statistic-rain_statistic.min())/(rain_statistic.max()-rain_statistic.min())*100+35
rain_color[:,:,1]=np.ones((1024,2048))*255
rain_color[:,:,2]=np.ones((1024,2048))*255
rain_color_2=np.array(rain_color,dtype=np.uint8)
rain_color_2=cv2.blur(rain_color_2,(3,3))
rain_color_2=cv2.cvtColor(rain_color_2,cv2.COLOR_HSV2BGR)
cv2.imwrite("visualize\\\\rain_color.jpg",rain_color_2)

#最后我们再画个图例
legend=np.zeros((10,100,3),dtype=np.uint8)
for j in range(100):
    legend[:,j,0]=np.ones((10),dtype=np.uint8)*j+35
legend[:,:,1]=np.ones((10,100),dtype=np.uint8)*255
legend[:,:,2]=np.ones((10,100),dtype=np.uint8)*255
legend_rgb=cv2.cvtColor(legend,cv2.COLOR_HSV2BGR)
cv2.imwrite("visualize\\\\legend_rain.jpg",legend_rgb)

legend=np.zeros((10,100,3),dtype=np.uint8)
for j in range(100):
    legend[:,j,0]=-np.ones((10),dtype=np.uint8)*j+100
legend[:,:,1]=np.ones((10,100),dtype=np.uint8)*255
legend[:,:,2]=np.ones((10,100),dtype=np.uint8)*255
legend_rgb=cv2.cvtColor(legend,cv2.COLOR_HSV2BGR)
cv2.imwrite("visualize\\\\legend_temperature.jpg",legend_rgb)
