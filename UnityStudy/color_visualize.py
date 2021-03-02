# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:39:28 2021

@author: wwp
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
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

rain_statistic=rain_statistic#*(1-is_water_matrix)
rain_color=np.zeros((1024,2048,3))#h从0到100，即红到蓝
rain_color[:,:,0]=(rain_statistic-rain_statistic.min())/(rain_statistic.max()-rain_statistic.min())*100+75
rain_color[:,:,1]=np.ones((1024,2048))*255
rain_color[:,:,2]=np.ones((1024,2048))*255
rain_color_2=np.array(rain_color,dtype=np.uint8)
rain_color_2=cv2.blur(rain_color_2,(3,3))
rain_color_2=cv2.cvtColor(rain_color_2,cv2.COLOR_HSV2BGR)
is_water_matrix=np.array(is_water_matrix,dtype=np.int)
rain_color_3=np.zeros((1024,2048,3))
rain_color_3[:,:,0]=rain_color_2[:,:,0]*(1-is_water_matrix)
rain_color_3[:,:,1]=rain_color_2[:,:,1]*(1-is_water_matrix)
rain_color_3[:,:,2]=rain_color_2[:,:,2]*(1-is_water_matrix)
cv2.imwrite("visualize\\\\rain_color_3.jpg",rain_color_3)

def cvt_temperature(t):
    T=(t-80.9)*(34.5+25)/(217.7-80.9)-25
    return T
def cvt_rain(r):
    R=r*8000/121.5
    return R
print("单位转换后的数据")
print("年平均气温 最高：",cvt_temperature(temperature_statistic.max())," 最低:",cvt_temperature(temperature_statistic.min())," 全球平均:",cvt_temperature(temperature_statistic.mean()))
print("年平均降雨 最高：",cvt_rain(rain_statistic.max())," 最低:",cvt_rain(rain_statistic.min())," 全球平均:",cvt_rain(rain_statistic.mean()))
print("左下角雪山山顶温度",cvt_temperature(temperature_statistic[733,98]))
print("左下角雪山山坡温度",cvt_temperature(temperature_statistic[734,151]))
print("中间偏下雪山山顶温度",cvt_temperature(temperature_statistic[685,1166]))
print("中间偏下雪山山坡温度",cvt_temperature(temperature_statistic[708,1160]))
print("中间赤道沙漠温度",cvt_temperature(temperature_statistic[530,1230]))

print("左下角雪山山顶降水量",cvt_rain(rain_statistic[733,98]))
print("左下角雪山山坡降水量",cvt_rain(rain_statistic[734,151]))
print("中间偏下雪山山顶降水量",cvt_rain(rain_statistic[685,1166]))
print("中间偏下雪山山坡降水量",cvt_rain(rain_statistic[708,1160]))
print("中间赤道沙漠降水量",cvt_rain(rain_statistic[530,1230]))

print("最大降水量位置",rain_statistic.argmax()/2048,rain_statistic.argmax()%2048)
print("最高温度位置",temperature_statistic.argmax()/2048,temperature_statistic.argmax()%2048)
print("最低温度位置",temperature_statistic.argmin()/2048,temperature_statistic.argmin()%2048)

path="unity.png"
terrain=cv2.imread(path)#等面积投影地图
cv2.circle(terrain, (int(temperature_statistic.argmax()%2048),int(temperature_statistic.argmax()/2048)), 5, (0,0,255), 3)#哪个nt设计的平时都用y,x坐标，这里画圈用x,y坐标
cv2.circle(terrain, (int(temperature_statistic.argmin()%2048),int(temperature_statistic.argmin()/2048)), 5, (255,0,0), 3)
cv2.circle(terrain, (int(rain_statistic.argmax()%2048),int(rain_statistic.argmax()/2048)), 5, (255,255,255), 3)
cv2.circle(terrain, (98,733), 3, (255,255,255), 3)
cv2.circle(terrain, (151,734), 3, (255,255,255), 3)
cv2.circle(terrain, (1166,685), 3, (255,255,255), 3)
cv2.circle(terrain, (1160,708), 3, (255,255,255), 3)
cv2.circle(terrain, (1230,530), 3, (255,255,255), 3)
cv2.imwrite("visualize\\\\hotspots.jpg",terrain)


#最后我们再画个图例
legend=np.zeros((10,100,3),dtype=np.uint8)
for j in range(100):
    legend[:,j,0]=np.ones((10),dtype=np.uint8)*j+75
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

#周期变化研究
global_mean_temperature_list=np.load("result\\\\temperature_list.npy")
global_mean_rain_list=np.load("result\\\\rain_list.npy")
plt.plot(np.arange(365),global_mean_temperature_list)
plt.show()
plt.plot(np.arange(365),global_mean_rain_list)
plt.show()