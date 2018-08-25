import numpy as np
from Figure_denoise import median_filter, crop, sobel, canny, bilateral_filter, top_hat_transform, bottom_hat_transform
import cv2
import numpy as np
# all picture

def cut_a_box(x_box_start,x_box_length,y_box_start, y_box_length ):
    """
    define a box, up left corner is the start point
    :param x_box_start:
    :param x_box_length:
    :param y_box_start:
    :param y_box_length:
    :return: the image region within the box
    """
    rect = img[y_box_start: (y_box_start + y_box_length), x_box_start:(x_box_start + x_box_length)]  # [y, x]
    return rect

def cal_mean_within_box(img):
    total = 0
    for i in range(np.shape(img)[0]):
        for j in range (np.shape(img)[1]):
            total += img[i, j]
    mean = total/float((np.shape(img)[0]*np.shape(img)[0]))
    #mean = total/float(2500.0)
    return mean

def first_the_first_level_rect(crop_length, x_box_length, y_box_length):
    rect_firt = []
    crop_length
    #x_box_start = 200
    x_box_length
    #y_box_start = 200
    y_box_length
   # mean = np.zeros((((crop_length - x_box_length)/10), ((crop_length - y_box_length)/10)))
    mean = np.zeros((50, 50))
    index_min_mean = np.zeros((14, 2))
    for picture in range(14):
        source = 'Data\Radiographs\%02d.tif' % (picture + 1)
        raw_img = cv2.imread(source, 0)
        cropped_img = crop(raw_img, 1000, 500, 1000, 1000)
        img = median_filter(cropped_img)
        img = bilateral_filter(img)
        #print raw_img
        #img = top_hat_transform(img)
        # img = bottom_hat_transform(img)
        img = sobel(img)
       # print (np.shape(img))
        for i in range(0, np.shape(img)[0] - x_box_length, 10):# x range search
            for j in range(0, np.shape(img)[1] - y_box_length, 10):# y range search
                #print (j)
                img_tmp = img[j: (j + y_box_length), i:(i + x_box_length)]
               # print (j, j+y_box_length, i, i + x_box_length)
                #print(i, j)
               #mean_tmp = float(cal_mean_within_box(img_tmp))
                mean_tmp = float(np.mean(img_tmp))
                #print(i,j)
                mean[i/10, j/10] = mean_tmp
        #print np.shape(mean)
        #print(picture)
        #print mean
        #print()
        index_min_mean[picture] = np.where(mean == np.amin(mean, axis = (0,1)))
        #print index_min_mean
        rect_tmp_for_show = img[int(index_min_mean[picture][0]) + 200: int((index_min_mean[picture][0]+200 + y_box_length)),
                           int(index_min_mean[picture][1])+ 200:int((index_min_mean[picture][1]+200 + x_box_length))]
        rect_firt.append((img[int(index_min_mean[picture][0]) + 200: int((index_min_mean[picture][0]+200 + y_box_length)),
                           int(index_min_mean[picture][1])+ 200:int((index_min_mean[picture][1]+200 + x_box_length))]))
        cv2.imshow('min_mean area', rect_tmp_for_show)
        cv2.waitKey(500)
        cv2.imwrite('Data\Configure\init_guess_image-%d.tif' % picture, rect_tmp_for_show)
    return rect_firt



if __name__ =='__main__':
    first_rect = first_the_first_level_rect(crop_length =1000, x_box_length = 500, y_box_length=500)




    #print index_min_mean





