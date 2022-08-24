import math
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import ImageEnhance,Image
from matplotlib import pyplot as plt

#检测车牌颜色
#参数：原始图片
#返回值：'蓝色','绿色','黄色'
def get_color(image):
    height=image.shape[0]
    width=image.shape[1]
    #设定阈值
    lower_blue=np.array([100,43,46])
    upper_blue=np.array([140,255,255])
    lower_yellow=np.array([15,55,55])
    upper_yellow=np.array([50,255,255])
    lower_green=np.array([20,30,116])
    upper_green=np.array([76,211,255])
    #转换为hsv
    hsv_img=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #根据阈值构建掩膜
    mask_blue=cv.inRange(hsv_img,lower_blue,upper_blue)
    mask_yellow=cv.inRange(hsv_img,lower_yellow,upper_yellow)
    mask_green=cv.inRange(hsv_img,lower_green,upper_green)
    #对mask进行操作--黑白像素点统计
    #记录黑白像素总和
    blue_white=0
    blue_black=0
    yellow_white=0
    yellow_black=0
    green_white=0
    green_black=0
    #记录每一列的黑白像素总和
    for i in range(width):
        for j in range(height):
            if mask_blue[j][i]==255:
                blue_white+=1
            if mask_blue[j][i]==0:
                blue_black+=1
            if mask_yellow[j][i]==255:
                yellow_white+=1
            if mask_yellow[j][i]==0:
                yellow_black+=1
            if mask_green[j][i]==255:
                green_white+=1
            if mask_green[j][i]==0:
                green_black+=1
    color_list=['蓝色','黄色','绿色']
    num_list=[blue_white,yellow_white,green_white]
    #返回白色区域最多的颜色
    return color_list[num_list.index(max(num_list))]

#根据hsv图像获取二值图像以得到轮廓
#参数：res图像
#返回值：二值化图像
def get_binary_image(image,color):
    #设定阈值
    lower_green = np.array([10, 30, 116])
    higher_green = np.array([76, 211, 255])
    lower_yellow = np.array([15, 55, 55])
    higher_yellow = np.array([55, 255, 255])
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    if color == '蓝色':
        # 掩膜：BGR通道，若像素B，G，R分量在范围内置255（白色），否则置0（黑色）
        mask_gbr = cv.inRange(image, (100, 0, 0), (255, 190, 140))
        # 转换成HSV颜色空间
        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # 分离通道，h（色调）,s（饱和度）,v（明度）
        h, s, v = cv.split(img_hsv)
        # 取饱和度通道进行掩膜取得二值图像
        mask_s = cv.inRange(s, 80, 255)
        # 与操作，两个二值图像都为白色才保留，否则置黑
        rgbs = mask_gbr & mask_s
        # 核的横向分量大，使车牌数字尽量连在一起
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 3))
        # 膨胀，减小车牌空洞
        img_rgbs_dilate = cv.dilate(rgbs, kernel, 3)
        # 进行开操作，去除细小噪点
        eroded = cv.erode(img_rgbs_dilate, None, iterations=1)
        img_rgbs_dilate = cv.dilate(eroded, None, iterations=1)
        # 去除噪点
        img_rgbs_dilate = cv.medianBlur(img_rgbs_dilate, 15)
        return img_rgbs_dilate
    if color == '绿色':
        mask = cv.inRange(hsv_image, lower_green, higher_green)
    if color == '黄色':
        mask = cv.inRange(hsv_image, lower_yellow, higher_yellow)
    res_image = cv.bitwise_and(image, image, mask=mask)
    # 高斯模糊
    gaussian = cv.GaussianBlur(src=res_image, ksize=(3, 3), sigmaX=0, sigmaY=0, borderType=cv.BORDER_DEFAULT)
    # 灰度化
    gray = cv.cvtColor(gaussian, cv.COLOR_BGR2GRAY)
    #闭操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    gray = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    # 使用SOBEL一阶导数方法
    sobel_grad_x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3)  # 对x方向求一阶导
    sobel_grad_y = cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=3)  # 对y方向求一阶导
    cv8u_grad_x = cv.convertScaleAbs(sobel_grad_x)  # 将计算结果转换成CV_8U像素类型
    cv8u_grad_y = cv.convertScaleAbs(sobel_grad_y)
    gray_image = cv.addWeighted(cv8u_grad_x, 0.5, cv8u_grad_y, 0.5, 0)  # 合并x和y方向的结果合成最终输出
    #二值化
    ret, binary_niblack = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY)
    # 闭合操作
    element = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    closed = cv.morphologyEx(binary_niblack, cv.MORPH_CLOSE, element, iterations=3)
    # 去除小白点
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 20))
    # 膨胀，腐蚀
    image = cv.dilate(closed, kernelX)
    image = cv.erode(image, kernelX)
    image = cv.erode(image, kernelY)
    image = cv.dilate(image, kernelY)
    # 去除噪点
    img_rgbs_dilate = cv.medianBlur(image, 15)
    return img_rgbs_dilate

#寻找有可能是车牌的轮廓
#参数：轮廓
#返回值：车牌区域列表，索引列表，形状列表
def findcontours(contours):
    # 找出最有可能是车牌的位置
    def getSatifyestBox(list_rate):
        for index, key in enumerate(list_rate):
            list_rate[index] = abs(key - 3)
        index = list_rate.index(min(list_rate))
        return index
    region=[]
    #车牌最小比例
    minPlateRatio=2.0
    #车牌最大比例
    maxPlateRatio=5.0
    #筛选面积
    list_rate=[]
    index_list=[]
    shape_list=[]
    for i in range(len(contours)):
        cnt=contours[i]
        #计算轮廓面积
        area=cv.contourArea(cnt)
        #面积筛选
        if area <1500:
            continue
        #转换为最小正交矩形
        rect=cv.minAreaRect(cnt)
        #根据矩形转成box类型，并int化
        box=np.int32(cv.boxPoints(rect))
        x,y,w,h=cv.boundingRect(cnt)
        #根据宽高比进行筛选
        ratio=float(w)/float(h)
        if ratio>maxPlateRatio or ratio <minPlateRatio:
            continue
        #车牌轮廓
        region.append(box)
        #存储宽高比
        list_rate.append(ratio)
        #车牌大小和定位
        shape_list.append([x,y,w,h])
        #符合要求的索引
        index=getSatifyestBox(list_rate)
        #存储符合要求的索引
        index_list.append(index)
    return region,index_list,shape_list

#旋转车牌
#参数：车牌轮廓，原始图像
#返回值：旋转后的图像
def rotate(contour,plate_image):
    h,w=plate_image.shape[:2]
    [vx,vy,x,y]=cv.fitLine(contour,cv.DIST_L2,0,0.01,0.01)
    #斜率
    k=vy/vx
    #截距
    b=y-k*x
    # lefty=b
    # righty=k*w+b
    # image=plate_image.copy()
    # line_image=cv.line(image,(int(w),int(righty)),(0,int(lefty)),(0,255,0),2)
    a=math.atan(k)
    #转成角度
    a=math.degrees(a)
    image=plate_image.copy()
    #放大
    M=cv.getRotationMatrix2D((w/2,h/2),a,0.8)
    #仿射变换
    dst=cv.warpAffine(image,M,(int(w*1.1),int(h*1.1)))
    return dst


#蓝色车牌锐化，二值化
#增加对比度
def con(img):
    h,w,ch=img.shape
    src2=np.zeros([h,w,ch],img.dtype)
    con=cv.addWeighted(img,1.2,src2,1-1.2,0)
    return con
#锐化
def rui(img):
    fil=np.array([[-1,-1,-1],[-1,9,-1],[-1,1,-1]])
    res=cv.filter2D(img,-1,fil)
    return res
def get_binary_bin(img,color):
    if color=='蓝色':
        rui_img=rui(img)
        #转换成HSV颜色空间
        img_hsv=cv.cvtColor(rui_img,cv.COLOR_BGR2HSV)
        #分离h,s,v通道
        h,s,v=cv.split(img_hsv)
        #二值化
        _,rui_otsu=cv.threshold(h,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        #增强对比度，取channel_s
        con_img=con(img)
        img_hsv=cv.cvtColor(con_img,cv.COLOR_BGR2HSV)
        h,s,v=cv.split(img_hsv)
        _,con_otsu=cv.threshold(s,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        #与操作，清晰，去噪
        con_rui=rui_otsu&con_otsu
        #二值化
        is_success,binary=cv.threshold(con_rui,0,255,cv.THRESH_OTSU)
        return binary
    if color=='绿色':
        b, g, r = cv.split(img)
        _, bin = cv.threshold(g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return bin
    if color=='黄色':
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, gray_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return gray_img

#字符分割
#参数：二值化图像
#返回值：字符列表
def char_seperator(bin_image,color):
    #记录分割字符的方法
    #0代表投影法
    type=0
    if color == '蓝色' or color == '黄色':
        segmentation_spacing = 0.95
    if color == '绿色':
        segmentation_spacing = 0.9
    #截去边框
    offset_X = 1
    offset_Y = 2
    offset_region = bin_image[offset_Y:-offset_Y, offset_X:-offset_X]
    image = offset_region
    # kernal = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))  # 至少为（5，）否则川连不上
    # image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernal)
    #未进行闭合操作版本
    image_distinct=offset_region.copy()
    #投影法
    def get_white_black(image,image_distinct):
        sorted_regions = []
        height = image.shape[0]
        width = image.shape[1]
        #统计黑白像素点
        def get_wb(image):
            white = []
            black = []
            height = image.shape[0]
            width = image.shape[1]
            white_max = 0
            black_max = 0
            #统计白色像素点和谁色像素点
            for i in range(width):
                w_count = 0
                b_count = 0
                for j in range(height):
                    if image[j][i] == 255:
                        w_count += 1
                    else:
                        b_count += 1
                white_max = max(white_max, w_count)
                black_max = max(black_max, b_count)
                white.append(w_count)
                black.append(b_count)
            return white, black, white_max, black_max

        white, black, white_max, black_max = get_wb(image)
        start_1 = 0
        end_1 = 0
        start_2 = 0
        end_2 = 0
        start_11=0
        #前面部分
        for i in range(20):
            if i == 0:
                continue
            #前面是大面积白色
            if white[0] >= 40:
                j = 0
                while white[j] >= 35:
                    j += 1
                end_2 = j
                break
            #突然出现一条白色（车牌边框）
            if abs(white[i] - white[i - 1]) > 30:
                start_1 = i
            #边框和车牌间的分界
            if abs(black[i]) > 40 and start_1 != 0:
                end_1 = i
                break
        #后面部分
        for i in range(width - 20, width):
            #字符结束
            if abs(white[i] - white[i - 1]) > 35:
                start_11 = i
            #后面部分大面积白色
            if white[i] > 45:
                j = 0
                while white[j] > 35:
                    j += 1
                if j>3:
                    start_2 = j
                break
        if end_1 != 0 :
            image = image[offset_Y:, end_1:]
            image_distinct = image_distinct[offset_Y:, end_1:]
            white, black, white_max, black_max = get_wb(image)
        if start_11!=0:
            image = image[:, :start_11]
            image_distinct = image_distinct[:, :start_11]
            white, black, white_max, black_max = get_wb(image)
        if end_2 != 0:
            image = image[offset_Y:, end_2:]
            image_distinct = image_distinct[offset_Y:, end_2:]
            if start_2 != 0:
                image = image[offset_Y:, :-start_2]
                image_distinct = image_distinct[offset_Y:,:-start_2]
            if start_11 != 0 and start_2>start_11:
                image = image[offset_Y:, :start_11]
                image_distinct = image_distinct[offset_Y:, :start_11]
            white, black, white_max, black_max = get_wb(image)
        if abs(black_max - white_max) < 2:
            image = image[offset_Y:-3, offset_X:-1]
            image_distinct = image_distinct[offset_Y:-3, offset_X:-1]
            white, black, white_max, black_max = get_wb(image)
        # 黑底白字
        image=image_distinct
        height = image.shape[0]
        width = image.shape[1]
        cv.waitKey(0)

        #找字符结束位置
        def find_end(start_):
            end_ = start_ + 1
            for m in range(start_ + 1, width - 1):
                wid=m-start_
                #是一个字符
                if wid<18:
                    #两个字符之间分界线或结尾
                    if (black[m] >= segmentation_spacing * black_max):
                        end_ = m
                        break
                #可能为两个字符，减小阈值
                else:
                    if (black[m] >= (segmentation_spacing * black_max-5)):
                        end_ = m
                        break
                #最后一部分
                if m == width - 2 and len(sorted_regions) < 8:
                    end_ = m
                    break
            return end_

        n = 1
        start = 1
        end = 2
        while n < width - 1:
            n += 1
            #每个字符开头
            if (white[n] >= (1 - segmentation_spacing-0.01) * white_max):
                start = n
                end = find_end(start)
                n = end
                #前两个字符不可能是数字一
                if len(sorted_regions)<2:
                    if end - start > 6:
                        cj = image[1:, start:end]
                        area = cj.shape[0] * cj.shape[1]
                        if area > 200:
                            sorted_regions.append(cj)
                else:
                    #分辨数字一
                    if end - start > 2:
                        cj = image[1:height, start:end]
                        area=cj.shape[0] * cj.shape[1]
                        white1, black1, white_max1, black_max1=get_wb(cj)
                        #统计该部分白色像素数，如果过小，就不是字符
                        white11=sum(white1)
                        if area > 100 and white11>50:
                            sorted_regions.append(cj)
        if color == '蓝色' or color == '黄色':
            if len(sorted_regions) < 7:
                sorted_regions = [0]
            elif len(sorted_regions) == 7:
                return sorted_regions
            else:
                sorted_regions = sorted_regions[:7]
        if color == '绿色':
            if len(sorted_regions) < 8:
                sorted_regions = [0]
            elif len(sorted_regions) == 8:
                return sorted_regions
            else:
                sorted_regions = sorted_regions[:8]
        return sorted_regions
    sorted_regions=get_white_black(image,image_distinct)
    if sorted_regions == [0]:
        image = offset_region
        image_distinct = offset_region
        kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 2))#至少为（5，）否则川连不上
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernal)
        sorted_regions=get_white_black(image,image_distinct)
    if sorted_regions==[0]:
        type=1
        # 字符分割
        # 向内缩进，去除外边框
        # 经验值
        offset_X = 3
        offset_Y = 5
        offset_region = bin_image[offset_Y:-offset_Y, offset_X:-offset_X]
        # 生成工作区域
        working_region = offset_region
        working_region=image_distinct.copy()
        # 对汉字区域进行等值线找区域
        # 经验值：汉子区域占整体的1/8
        chinese_char_max_width = working_region.shape[1] // 8
        # 提取汉字区域
        chinese_char_region = working_region[:, 0:chinese_char_max_width]
        # # 对汉字区域进行模糊处理
        #cv.GaussianBlur(chinese_char_region, (9, 9), 0, dst=chinese_char_region)
        # 对整个区域找轮廓--等值线
        image = working_region.copy()
        kernal = cv.getStructuringElement(cv.MORPH_RECT, (3, 2))
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernal)
        char_contours, _ = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        cv.drawContours(image, char_contours, 0, (0, 255, 0), 2)
        working_region=offset_region
        # 过滤不合适的轮廓
        # 经验值
        CHAR_MIN_WIDTH = working_region.shape[1] // 40
        CHAR_MIN_HEIGHT = working_region.shape[0] * 7 // 10

        # 逐个遍历所有候选的字符区域轮廓==等值线框，按照大小进行过滤
        valid_char_regions = []
        for i in np.arange(len(char_contours)):
            x, y, w, h = cv.boundingRect(char_contours[i])
            if w >= CHAR_MIN_WIDTH and h >= CHAR_MIN_HEIGHT:
                # 将字符区域的中心店x的坐标和字符区域作为一个元组，放入列表
                valid_char_regions.append((x, offset_region[y:y + h, x-2:x + w]))
        # 按照区域的x坐标进行排序，并返回字符列表
        sorted_regions = sorted(valid_char_regions, key=lambda region: region[0])
        if color == '蓝色' or color == '黄色':
            if len(sorted_regions) < 7:
                sorted_regions = [0]
            elif len(sorted_regions) == 7:
                return sorted_regions,type
            else:
                sorted_regions = sorted_regions[:7]
        if color == '绿色':
            if len(sorted_regions) < 8:
                sorted_regions = [0]
            elif len(sorted_regions) == 8:
                return sorted_regions,type
            else:
                sorted_regions = sorted_regions[:8]
    return sorted_regions,type

