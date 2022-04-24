'''

'''

import numpy as np
import math

class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y**2 + self.x**2))
        self.step = step
        self.vote_matrix = np.zeros([math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print ('Hough_transform_algorithm')
        # ------------- write your code bellow ----------------
        self.angle = np.tan(self.angle)
        for i in range(0,self.y):
            for j in range(0,self.x):
                if self.img[i][j]<=0:
                    continue
                #方向1
                r=0
                x=j
                y=i
                while y<self.y and x<self.x and y>=0 and x>=0:
                    self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                        math.floor(r / self.step)] += 1
                    x+=self.step
                    y+=self.step*self.angle[i][j]
                    r+=math.sqrt((self.step*self.angle[i][j])**2+self.step**2)
                #方向2
                x=j-self.step
                y=i-self.step*self.angle[i][j]
                r=math.sqrt((self.step*self.angle[i][j])**2+self.step**2)
                while y<self.y and x<self.x and y>=0 and x>=0:
                    self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                        math.floor(r / self.step)] += 1
                    x-=self.step
                    y-=self.step*self.angle[i][j]
                    r+=math.sqrt((self.step*self.angle[i][j])**2+self.step**2)
        # ------------- write your code above ----------------
        return self.vote_matrix


    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print ('Select_Circle')
        # ------------- write your code bellow ----------------
        possible_circle=[]
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i * self.step + self.step / 2
                        x = j * self.step + self.step / 2
                        radius = r * self.step + self.step / 2
                        vote_num=self.vote_matrix[i][j][r]
                        possible_circle.append((math.ceil(x), math.ceil(y), math.ceil(radius),vote_num))
        if len(possible_circle) == 0:
            print("No circle in this threshold")
            return

        #此时possible_circle是先按照y排序，后按照x排序
        possible_circle.sort(key=lambda x: x[3])#按照投票数目排序
        flag=[1]*len(possible_circle)
        for i in range(len(possible_circle)):
            if flag[i]==0: continue
            circle=possible_circle[i]
            x=circle[0]
            y=circle[1]
            r=circle[2]
            sum=circle[3]
            for j in range(i+1,len(possible_circle)):
                if flag[j] == 0: continue
                temp=possible_circle[j]
                if abs(x - temp[0]) <= 20 and abs(y - temp[1]) <= 20:
                    sum += temp[3]
                    flag[j]=2
            summ=1.0/sum
            print(sum,summ)
            x = circle[0]*circle[3]*summ
            y = circle[1] * circle[3] * summ
            r = circle[2] * circle[3] * summ
            for j in range(i+1,len(possible_circle)):
                if flag[j] == 0: continue
                temp=possible_circle[j]
                if flag[j]==2:
                    flag[j]=0
                    x += temp[0] * temp[3] * summ
                    y += temp[1] * temp[3] * summ
                    r += temp[2] * temp[3] * summ
            print(x,y,r)
            self.circles.append((x, y, r))
        # ------------- write your code above ----------------


    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles