'''

'''
import math
import queue

import cv2
import numpy as np
import que as que


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold, Save_path):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Save_Path = Save_path
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        已经经历过高斯滤波平滑
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print('Get_gradient_img')
        # ------------- write your code bellow ----------------
        gradient_x = np.zeros([self.y, self.x], dtype=np.float)
        gradient_y = np.zeros([self.y, self.x], dtype=np.float)
        x_length = self.x
        y_length = self.y
        for i in range(0, self.y):
            for j in range(0, self.x):
                gradient_x[i][j] = np.sum(
                    np.array([self.img[i][j], self.img[i][(j + 1) % x_length]]) * self.x_kernal)  # "*"乘法对应相乘
                gradient_y[i][j] = np.sum(
                    np.array([[self.img[i][j]], [self.img[(i + 1) % y_length][j]]]) * self.y_kernal)

        # 求梯度和角度的简单写法：使用cv2.cartToPolar将直角坐标系转为极坐标系，其中梯度=极径=sqrt(gradent_x^2+gradent_y^2)
        gradient_img, self.angle = cv2.cartToPolar(gradient_x, gradient_y)
        self.img = gradient_img.astype(np.uint8)

        # 输出测试：
        x_max = gradient_x[0][0]
        y_max = gradient_y[0][0]
        g_max = 0.0
        gradient = np.zeros([self.y, self.x], dtype=np.float)
        for i in range(0, self.y):
            for j in range(0, self.x):
                gradient_x[i][j] = gradient_x[i][j] * gradient_x[i][j]
                x_max = max(x_max, gradient_x[i][j])
                gradient_y[i][j] = gradient_y[i][j] * gradient_y[i][j]
                y_max = max(y_max, gradient_y[i][j])
                gradient[i][j] = (gradient_x[i][j] + gradient_y[i][j]) ** 0.5  # 用"**"来表示^
                g_max = max(g_max, gradient[i][j])

        for i in range(0, self.y):
            for j in range(0, self.x):
                gradient_x[i][j] /= x_max
                gradient_x[i][j] *= 255.0
                gradient_y[i][j] /= y_max
                gradient_y[i][j] *= 255.0
                gradient[i][j] /= g_max
                gradient[i][j] *= 255.0
        cv2.imwrite(self.Save_Path + "x_gradient.jpg", gradient_x.astype(np.uint8))
        cv2.imwrite(self.Save_Path + "y_gradient.jpg", gradient_y.astype(np.uint8))
        cv2.imwrite(self.Save_Path + "gradient.jpg", gradient.astype(np.uint8))

        # ------------- write your code above ----------------
        return self.img

    def get_gradient(self, x, y):
        '''
        输出(x,y)处梯度值
        '''
        if x < 0: x = 0
        if x > self.x - 1: x = self.x - 1
        if y < 0: y = 0
        if y > self.y - 1: y = self.y - 1
        x1 = math.floor(x)
        x2 = math.ceil(x)
        y1 = math.floor(y)
        y2 = math.ceil(y)
        r = 0.0

        if x1 == x2 and y1 == y2:
            r = self.img[y1][x1]
        elif x1 == x2:
            r = self.img[y1][x1] * (1 - y + y1) + self.img[y2][x1] * (y - y1)
        elif y1 == y2:
            r = self.img[y1][x1] * (1 - x + x1) + self.img[y1][x2] * (x - x1)  # 两个像素不能相减会出现负值！！！！！！！！！
        else:
            print("case:3")
        return r

    def Non_maximum_suppression(self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print('Non_maximum_suppression')
        # print(self.x,self.y)
        # ------------- write your code bellow ----------------
        result = np.zeros([self.y, self.x])
        for i in range(0, self.y):
            for j in range(0, self.x):
                # if self.img[i][j] < self.HT_high_threshold:
                if self.img[i][j] <= 4:####################################################这里需要是4，也就是说高门槛是4，那么低门槛...?
                    result[i][j] = 0
                    continue
                else:
                    tangle = math.tan(self.angle[i][j])
                    if abs(tangle) < 1:
                        delta_x = 1
                        delta_y = tangle
                    else:
                        delta_y = 1
                        delta_x = 1.0 / tangle
                    # 方向1
                    y = i + delta_y
                    x = j + delta_x
                    temp1 = self.get_gradient(x, y)
                    # 方向2
                    y = i - delta_y
                    x = j - delta_x
                    temp2 = self.get_gradient(x, y)
                    if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                        result[i][j] = self.img[i][j]
                    else:
                        result[i][j] = 0
        self.img = result
        # ------------- write your code above ----------------

        # 输出测试：
        g_max = 0.0
        for i in range(0, self.y):
            for j in range(0, self.x):
                g_max = max(g_max, result[i][j])

        for i in range(0, self.y):
            for j in range(0, self.x):
                result[i][j] /= g_max
                result[i][j] *= 255.0
        cv2.imwrite(self.Save_Path + "Non_maximum_suppression.jpg", result.astype(np.uint8))

        return self.img

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print('Hysteresis_thresholding')
        # ------------- write your code bellow ----------------
        que = queue.Queue()
        for i in range(0, self.y):
            for j in range(0, self.x):
                if self.img[i][j] >= self.HT_high_threshold:
                    que.put(np.array([i, j]))
                    #print("i,j:",i,j)

        while que.qsize():
           # print(que.qsize())
            u = que.get()
            i = u[0]
            j = u[1]
           # print("u",u)
            angle = self.angle[i][j] + math.pi * 0.5
            tangle = math.tan(angle)
            if abs(tangle) < 1:
                if tangle > 0:
                    if j + 1 < self.x:
                        cnt=0 #记录该方向上未被更新的点数量(0,1,2)
                        ii=jj=-2
                        if self.img[i][j + 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i][j + 1] >= self.HT_low_threshold:
                                ii=0
                                jj=1
                        if i - 1 > 0 and self.img[i - 1][j + 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i - 1][j + 1] >= self.HT_low_threshold and self.img_origin[i - 1][j + 1]>self.img_origin[i][j + 1]:
                                ii=-1
                                jj=1
                        if cnt==2 and ii!=-2:
                            self.img[i+ii][j+jj] = max(self.HT_high_threshold, self.img_origin[i+ii][j+jj])
                            que.put(np.array([i+ii, j +jj]))
                    if j - 1 > 0:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i][j - 1]<self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i][j - 1] >= self.HT_low_threshold:
                                ii=0
                                jj=-1
                        if i + 1 < self.y and self.img[i + 1][j - 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][j - 1] >= self.HT_low_threshold and self.img_origin[i + 1][j - 1]>self.img_origin[i][j - 1]:
                                ii=1
                                jj=-1
                        if cnt==2 and ii!=-2:
                            self.img[i+ii][j+jj] = max(self.HT_high_threshold, self.img_origin[i+ii][j+jj])
                            que.put(np.array([i+ii, j +jj]))
                else:
                    if j + 1 < self.x:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i][j + 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i][j + 1] >= self.HT_low_threshold:
                                ii=0
                                jj=1
                        if i+1<self.y and self.img[i + 1][j + 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][j + 1] >= self.HT_low_threshold and self.img_origin[i + 1][j + 1] >self.img_origin[i][j + 1]:
                                ii=1
                                jj=1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))
                    if j - 1 > 0:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i][j - 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i][j - 1] >= self.HT_low_threshold:
                                ii=0
                                jj=-1
                        if i-1>0 and self.img[i - 1][j - 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i - 1][j - 1] >= self.HT_low_threshold and self.img_origin[i - 1][j - 1] >self.img_origin[i][j - 1]:
                                ii=-1
                                jj=-1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))
            else:
                if tangle > 0:
                    if i + 1 < self.y:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i + 1][j] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][j] >= self.HT_low_threshold:
                                ii=1
                                jj=0
                        if j - 1 > 0 and self.img[i + 1][j - 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][ j - 1] >= self.HT_low_threshold and self.img_origin[i + 1][ j - 1] >self.img_origin[i + 1][j]:
                                ii=1
                                jj=-1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))
                    if i - 1 > 0:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i - 1][j] < self.HT_high_threshold:
                            cnt += 1
                            if self.img_origin[i - 1][j] >= self.HT_low_threshold:
                                ii = -1
                                jj = 0
                        if j + 1 <self.x and self.img[i - 1][j + 1] < self.HT_high_threshold:
                            cnt += 1
                            if self.img_origin[i - 1][j + 1] >= self.HT_low_threshold and self.img_origin[i - 1][
                                j + 1] > self.img_origin[i - 1][j]:
                                ii = -1
                                jj = 1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))
                else:
                    if i + 1 < self.y:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i + 1][j] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][j] >= self.HT_low_threshold:
                                ii=1
                                jj=0
                        if j + 1 < self.x and self.img[i + 1][j + 1] < self.HT_high_threshold:
                            cnt+=1
                            if self.img_origin[i + 1][ j + 1] >= self.HT_low_threshold and self.img_origin[i + 1][ j + 1] >self.img_origin[i + 1][j]:
                                ii=1
                                jj=1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))
                    if i - 1 > 0:
                        cnt = 0  # 记录该方向上未被更新的点数量(0,1,2)
                        ii = jj = -2
                        if self.img[i - 1][j] < self.HT_high_threshold:
                            cnt += 1
                            if self.img_origin[i - 1][j] >= self.HT_low_threshold:
                                ii = -1
                                jj = 0
                        if j - 1 > 0 and self.img[i - 1][j - 1] < self.HT_high_threshold:
                            cnt += 1
                            if self.img_origin[i - 1][j - 1] >= self.HT_low_threshold and self.img_origin[i - 1][
                                j - 1] > self.img_origin[i - 1][j]:
                                ii = -1
                                jj = -1
                        if cnt == 2 and ii != -2:
                            self.img[i + ii][j + jj] = max(self.HT_high_threshold, self.img_origin[i + ii][j + jj])
                            que.put(np.array([i + ii, j + jj]))

        return self.img


    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''
        # 1.高斯滤波
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        # 2.求导(梯度)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        # 3.非最大化抑制
        self.Non_maximum_suppression()
        # 4.Hysteresis thresholding
        self.Hysteresis_thresholding()
        return self.img
