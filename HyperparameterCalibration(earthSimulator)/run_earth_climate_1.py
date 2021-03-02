# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:25:01 2021#主要是28号在火车上写的

@author: Templar_KBCC
"""

import numpy as np
import os
import cv2
#读取初始化值：
path="earth.jpg"
terrain=cv2.imread(path)#等面积投影地图
mapshape=[1024,2048]
temperature_matrix=np.load("result\\\\temperature_matrix.npy")
#temperature_matrix=temperature_matrix+273.15#转换为开尔文
humidity_matrix=np.load("result\\\\humidity_matrix.npy")
humidity_generate_matrix=np.load("initial\\\\humidity_generate_matrix.npy")
altitude_matrix=np.load("initial\\\\altitude_matrix.npy")
water_exhaustion_matrix=np.load("initial\\\\water_exhaustion_matrix.npy")
terrain_heat_absorb_matrix=np.load("initial\\\\terrain_heat_absorb_matrix.npy")
terrain_heat_capacity_matrix=np.load("initial\\\\terrain_heat_capacity_matrix.npy")#热惯性矩阵，水域应该温度变化有明显滞后
sun_heat_impact_matrix=np.load("initial\\\\sun_heat_impact_matrix.npy")#现在还没有，比较复杂，单独写个
is_water_matrix=np.load("initial\\\\is_water_matrix.npy")
#程序运算这个，365天都要，其实对一年的气候来讲，昼夜变化影响也许可以忽略，咱就只做一个一天变化的，以一天
#为基本单位
#province_size_matrix=np.ones(mapshape)我发现是等面积投影，所以不需要了
parameter_transfer_matrix=np.zeros(mapshape)#不需要初始化，只是留个坑
saturation_humidity_matrix=np.ones(mapshape)#不需要初始化，只是留个坑
surplus_humidity=np.ones(mapshape)
cloudy_generate_portion=0.000#我瞎猜的数#现在线性函数不对，有大于1的云度会出bug，改完再搞云度的问题
outerspace_radiation_decay=0.01#0.01的话30摄氏度，约300开尔文，一小时纯释放后变成197，降3度，略快
direct_sun_heat_impact=1.5#我瞎写的数
rain_water_exhaustion_multiplier=0.1
air_density=1.293#后面得用伯努利公式，不行啊，咱不用它了，用下一行这个impact参数模拟
pressure_difference_impact=0.2#也许这个得大一点才能散去特高温度？好像大了之后下雨就会非常集中在山区，因为平原攒不起来
altitude_transfer_multiplier=0.0001#山区和平原同温度，空气流向山区，因为山区海拔高，所以
#计算的时候，认为山区的温度较高，空气就会流进来
temperature_to_humidity_capacity_factor=0.01#300开尔文多容纳30的湿度
altitude_to_humidity_capacity_factor=-0.01#100米的山容纳量-0.1，改成山上饱和度上限更高，让降雨留在平原（虽然违反物理规律），然后为了防止山上的湿度太大，让山地水分衰减参数变大，这样还模拟了水流下山的过程，并且让山成为阻挡水汽的东西，就很合理
base_humidity_capacity=15#防止山地太高搞出负的湿度容纳量，山地设定最高就是1500
humidity_to_rain_factor=10
additional_rain_in_water_region_factor=5
transfer_kernel=np.array([[0.1,1,0.1],#地转偏向力也忽略
                 [1, 0,  1.0],#没有办法，为了计算速度快，只能忽略纵向横向接触面积不同的问题，大概记上地球自转影响吧
                 [0.1,1,0.1]])/4.4#不除以这个转移来转移去温度越来越他妈高，湿度也越来越他妈高
#地图是1024x2048,1:2，也就意味着经度方向和赤道方向单位像素对应实际长度是相同的,一像素约20km
rain_statistic=np.zeros(mapshape)
humidity_statistic=np.zeros(mapshape)
temperature_statistic=np.zeros(mapshape)
days=365
for i in range(days):
    print("day ",i)
    #第一部分：计算太阳照射和地表热辐射对温度的影响，他们都受到云度的影响（云度影响删掉，需要非线性函数，网上查不到很好的资料）
    anti_cloudy_degree_matrix=humidity_matrix*(1-cloudy_generate_portion)#晴朗程度矩阵
    tmp_sun_heat_impact_matrix=sun_heat_impact_matrix[i].reshape(1024,1)
    sun_heat_adding_matrix=tmp_sun_heat_impact_matrix*direct_sun_heat_impact/100*terrain_heat_absorb_matrix#一行的impact矩阵应该会自动广播，转置才能广播，因为读取得到的是行，而我们需要列
    heat_loss_matrix=temperature_matrix*outerspace_radiation_decay
    radiation_cause_temperature_change=(sun_heat_adding_matrix-heat_loss_matrix)#*anti_cloudy_degree_matrix这地方问题太大了，之前湿度越大直接温度变化幅度超大
    #第二部分：计算气象参数转移矩阵,是一个加权矩阵，给后面热量和湿度转移计算用的
        #温度越低的地区的参数对临近地区影响越大，但如果按这样算，暖空气就永远不会移动到冷空气的地方，所以
        #咱还是周围地区加权取平均吧。。。。。。。。。
    #parameter_transfer_matrix=cv2.filter2D(temperature_matrix,-1,transfer_kernel)#边值还没有处理
    #不要了，特么直接后面卷积得了，我好像想的有问题
    #第三部分：计算地形对湿度的影响（蒸发的增加与损失）
    prime_humidity_loss=humidity_matrix*water_exhaustion_matrix#oh shit，之前地形分析程序里是这东西乘下完事而不是减，怪不得沙漠降雨量最大
    terrain_based_humidity_change=humidity_generate_matrix*tmp_sun_heat_impact_matrix/100-prime_humidity_loss
    #第四部分：计算热量的转移与湿度的转移,暂时不考虑地图最东边最西边其实是连着的
    weighted_temperature_matrix=(altitude_matrix*altitude_transfer_multiplier+1)*temperature_matrix
    weighted_humidity_matrix=(altitude_matrix*altitude_transfer_multiplier+1)*humidity_matrix
    #用opencv的库函数需要先转成opencv的格式，opencv是有浮点数据类型矩阵的，em..实际上是后面transfer_kernel没用numpy错了而不是前面。。
    heat_transfer_matrix=cv2.filter2D(temperature_matrix,-1,transfer_kernel)#不用weighted试试
    humidity_transfer_matrix=cv2.filter2D(humidity_matrix,-1,transfer_kernel)
    #第五部分：计算上述影响叠加后下一时刻的温度、湿度矩阵
    temperature_matrix=temperature_matrix*(1-pressure_difference_impact/terrain_heat_capacity_matrix) \
        +heat_transfer_matrix*pressure_difference_impact/terrain_heat_capacity_matrix+radiation_cause_temperature_change/terrain_heat_capacity_matrix#热惯性
    humidity_matrix=humidity_matrix*(1-pressure_difference_impact) \
        +humidity_transfer_matrix*pressure_difference_impact+terrain_based_humidity_change
    #第六部分：计算水汽饱和矩阵（与温度和高度有关）
    saturation_humidity_matrix=altitude_matrix*altitude_to_humidity_capacity_factor+ \
        temperature_matrix*temperature_to_humidity_capacity_factor+base_humidity_capacity
    #第七部分：判断降雨地区并对降雨地区乘rain_water_exhaustion_multiplier，衰减部分计入降水统计
        #并二次更新湿度矩阵
    surplus_humidity=humidity_matrix-saturation_humidity_matrix#我他娘的反了，怪不得沙漠一直下雨
    ret,rain_determinant_matrix=cv2.threshold(surplus_humidity,1,1,cv2.THRESH_BINARY)#下雨的是1，不下是0，二值图像，不能用来计算
#上面这个ret好像是判断操作是否成功，没有的话这个matrix就是个二元素列表。。。第一个元素是1，第二个元素才是咱要的矩阵
    #这样自动下雨的地方会降雨，不下雨的地方湿度变化量为0
    rain_cause_humidity_change=humidity_matrix*rain_determinant_matrix* \
        rain_water_exhaustion_multiplier#这个被赋值的就是降水量
    humidity_matrix=humidity_matrix-rain_cause_humidity_change
    #第八部分：更新温度统计、湿度统计、降雨统计
    rain_statistic=rain_statistic+(rain_cause_humidity_change/days)*humidity_to_rain_factor+(rain_cause_humidity_change/days)*is_water_matrix*additional_rain_in_water_region_factor
    humidity_statistic=humidity_statistic+humidity_matrix/days
    temperature_statistic=temperature_statistic+temperature_matrix/days
    #(287,1689):北京
    #(339,1496):喜马拉雅山某处
    #(354,1077):非洲撒哈拉沙漠某处
#结束后保存最后时刻的湿度、温度矩阵、三个统计矩阵以供研究
np.save("result\\\\temperature_matrix",temperature_matrix)#至少保证以开尔文零度为标准零度
np.save("result\\\\humidity_matrix",humidity_matrix)
np.save("result\\\\rain_statistic",rain_statistic)
np.save("result\\\\humidity_statistic",humidity_statistic)
np.save("result\\\\temperature_statistic",temperature_statistic)
print("precipitation:")
print(rain_statistic[287,1689])
print(rain_statistic[339,1496])
print(rain_statistic[354,1077])
print("humidity:")
print(humidity_statistic[287,1689])
print(humidity_statistic[339,1496])
print(humidity_statistic[354,1077])
print("temperature:")
print(temperature_statistic[287,1689])
print(temperature_statistic[339,1496])
print(temperature_statistic[354,1077])
#待改进：水分蒸发速度应该从赤道递减，所以应该乘以sun_heat_impact/100，搞定之后反而看起来更离谱了。。先去掉
#待改进：地图东西两边是连着的
#太阳照射矩阵只有一年，天数大于365的话你得加上天数取模运算