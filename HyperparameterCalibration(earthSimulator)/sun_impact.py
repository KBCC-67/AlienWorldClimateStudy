# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 00:01:54 2021#火车上

@author: Templar_KBCC
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
mapshape=[1024,2048]
direct_sun_heat_impact=1.0#因为节省存储空间，这里我们不用2，防止溢出
days_of_year=365
sun_heat_impact_matrix=np.ones((days_of_year,mapshape[0]),dtype=np.int8)#即便是int8全存下来也要约0.7GB，所以咱还是省着用一行存一次
print(sun_heat_impact_matrix.size)

for day in range(days_of_year):
    sin_of_declination_angle=0.39795*m.cos(0.98563*(day-173)/180*m.pi)#网上抄的公式，也不知道是不是满足北纬为正
    cos_of_declination_angle=m.cos(m.asin(sin_of_declination_angle))#会有正负问题，所以不用sin^2+cos^2=1来推导
    for y in range(mapshape[0]):
        latitude=(y-mapshape[0]*0.5)*0.5*m.pi/(mapshape[0]*0.5)#0~512压缩到0~(1/2)pi，因为是弧度
        #北纬为正，我没有考虑到这是个等面积投影地图，只是进行了线性映射，有时间再推更准确的公式
        sin_of_solar_altitude_angle=m.sin(latitude)*sin_of_declination_angle+ \
            m.cos(latitude)*cos_of_declination_angle
        #cos_of_solar_altitude_angle=m.cos(m.asin(sin_of_solar_altitude_angle))
        impact=sin_of_solar_altitude_angle*direct_sun_heat_impact*100#画图发现是sin不需要cos
        if impact<=0:
            impact=0#极夜问题，什么？你说极昼？那不是该模型自动满足条件？
        sun_heat_impact_matrix[day][y]=impact
np.save("initial\\\\sun_heat_impact_matrix",sun_heat_impact_matrix)
print("day 0:")
plt.plot(np.arange(1024),sun_heat_impact_matrix[0])
plt.show()
print("day 90:")
plt.plot(np.arange(1024),sun_heat_impact_matrix[90])
plt.show()
print("day 180:")
plt.plot(np.arange(1024),sun_heat_impact_matrix[180])
plt.show()
print("day 270:")
plt.plot(np.arange(1024),sun_heat_impact_matrix[270])
plt.show()
#用的时候还要除以100，这里是存的小数点后两位