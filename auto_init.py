import numpy as np
from Figure_denoise import median_filter, crop, sobel, canny, bilateral_filter, top_hat_transform, bottom_hat_transform
import cv2
import numpy as np
from pre_process import Landmarks, rescale_withoutangle
from util import load_training_data
from iteration import active_shape_model, evaluation
from GPA import gpa
from Plotter import drawlines

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
    return mean

def first_the_first_level_rect(crop_length, x_box_length, y_box_length):
    """

    :param crop_length:
    :param x_box_length:
    :param y_box_length:
    :return: the min block top left point index from the ORIGINAL image
    """
    off_set_y = 200
    off_set_x = 240
    rect_firt = []
    crop_length
    #x_box_start = 200
    x_box_length
    #y_box_start = 200
    y_box_length
    mean = np.zeros(((crop_length - x_box_length)/10, (crop_length - y_box_length)/10))
    index_min_mean = np.zeros((14, 2))
    real_index_min_mean = np.zeros((14, 2))
    for picture in range(14):
        source = 'Data\Radiographs\%02d.tif' % (picture + 1)
        raw_img = cv2.imread(source, 0)
        cropped_img = crop(raw_img, 1000, 500, 1000, 1000)
        img = median_filter(cropped_img)
        img = bilateral_filter(img)
        img = top_hat_transform(img)
        img = bottom_hat_transform(img)
        img = sobel(img)

        for i in range(0, np.shape(img)[0] - x_box_length, 10):# x range search
            for j in range(0, np.shape(img)[1] - y_box_length, 10):# y range search
                img_tmp = img[j: (j + y_box_length), i:(i + x_box_length)]
                mean_tmp = float(np.mean(img_tmp))
                mean[i/10, j/10] = mean_tmp

        index_min_mean[picture] = np.where(mean == np.amin(mean, axis = (0,1)))
        real_index_min_mean[picture][0] = index_min_mean[picture][0] + off_set_x
        real_index_min_mean[picture][1] = index_min_mean[picture][1] + off_set_y
        rect_tmp_for_show = img[int(real_index_min_mean[picture][1]): int(real_index_min_mean[picture][1] + y_box_length),
                                int(real_index_min_mean[picture][0]):int((real_index_min_mean[picture][0] + x_box_length))]
        rect_firt.append((img[int(real_index_min_mean[picture][1]): int(real_index_min_mean[picture][1] + y_box_length),
                              int(real_index_min_mean[picture][0]):int((real_index_min_mean[picture][0] + x_box_length))]))
       # cv2.imshow('min_mean area', rect_tmp_for_show)
        #cv2.waitKey(500)
        cv2.imwrite('Data\Configure\init_guess_image-%d.tif' % picture, rect_tmp_for_show)
    return rect_firt, real_index_min_mean

def the_second_level_rect(rect,x_box_length, y_box_length, min_index_for_first_level):
    off_set_y = 150
    off_set_x = 40
    rect_second = []
    index_min_mean = np.zeros((14, 2))
    real_index_min_mean = np.zeros((14, 2))
    mean = np.zeros((int((np.shape(rect)[1] - x_box_length)/10), int((np.shape(rect)[2] - y_box_length)/10)))
    for picture in range(np.shape(rect)[0]):
        for i in range(0, np.shape(rect)[1] - x_box_length, 10):  # x range search
            for j in range(0, np.shape(rect)[2] - y_box_length, 10):  # y range search
                img_tmp = rect[picture][j: (j + y_box_length), i:(i + x_box_length)]
                mean_tmp = float(np.mean(img_tmp))
                mean[i / 10, j / 10] = mean_tmp
        index_min_mean[picture] = np.where(mean == np.amin(mean, axis=(0, 1)))
        real_index_min_mean[picture][0] = index_min_mean[picture][0] + off_set_x
        real_index_min_mean[picture][1] = index_min_mean[picture][1] + off_set_y
        rect_tmp_for_show = rect[picture][int(real_index_min_mean[picture][1]): int(real_index_min_mean[picture][1] + y_box_length),
                                int(real_index_min_mean[picture][0]):int((real_index_min_mean[picture][0] + x_box_length))]
        rect_second.append((rect[picture][int(real_index_min_mean[picture][1]): int(real_index_min_mean[picture][1] + y_box_length),
                              int(real_index_min_mean[picture][0]):int((real_index_min_mean[picture][0] + x_box_length))]))
       # cv2.imshow('min_mean area', rect_tmp_for_show)
       # cv2.waitKey(500)
        cv2.imwrite('Data\Configure\init_guess_second_level-%d.tif' % picture, rect_tmp_for_show)
    return rect_second, real_index_min_mean

def find_the_median_point_of_rect(rect):
    median_point = np.zeros((np.shape(rect)[0], 2))
    for picture in range(np.shape(rect)[0]):
        median_point[picture] = [np.shape(rect)[1]/2, np.shape(rect)[2]/2]
    return median_point


if __name__ =='__main__':
    x_box_length = 500
    y_box_length = 500
    first_rect ,min_index_for_first_level= first_the_first_level_rect(crop_length =1000, x_box_length = 500, y_box_length=500)
    second_rect, min_index_for_second_level = the_second_level_rect(first_rect, 440, 200, min_index_for_first_level)
    real_start_point = min_index_for_first_level + min_index_for_second_level
    auto_ini_pos = []
    x_00 = np.shape(14)
    x_01 = np.shape(14)
    x_02 = np.shape(14)
    x_03 = np.shape(14)

    y_0 = np.shape(14)
    y_1 = np.shape(14)


    print np.shape(real_start_point) # since here it is right
    final_middle_point = np.zeros((14, 2))
    """This is the final result of middle points"""
    for picture in range(14):
        final_middle_point[picture] = [real_start_point[picture, 0] + 220, real_start_point[picture, 1] + 100]
    for picture in range(14):
        source = 'Data\Radiographs\%02d.tif' % (picture + 1)
        raw_img = cv2.imread(source, 0)
        cropped_img = crop(raw_img, 1000, 500, 1000, 1000)
        img = median_filter(cropped_img)
        img = bilateral_filter(img)
        img = top_hat_transform(img)
        img = bottom_hat_transform(img)
        img = sobel(img)
        #cv2.circle(img, (int(final_middle_point[picture, 0]), int(final_middle_point[picture, 1])), 10, (255, 0, 0), 1)
        #cv2.imshow('center circle', img)
        #cv2.waitKey(500)
        #cv2.imwrite('Data\Configure\init_guess_center_point-%d.tif' % picture, img)

        x_length_top = 400
        x_length_bot = 300
        y_length = 800
        for i in range(14):
            x_00 = int(final_middle_point[i, 0] - (x_length_top/10)*4)
            x_01 = int(final_middle_point[i, 0] - (x_length_top/10)*1.5)
            x_02 = int(final_middle_point[i, 0] + (x_length_top/10)*1.5)
            x_03 = int(final_middle_point[i, 0] + (x_length_top/10)*4)
            x_10 = int(final_middle_point[i, 0] - (x_length_bot/10)*4)
            x_11 = int(final_middle_point[i, 0] - (x_length_bot/10)*1.5)
            x_12 = int(final_middle_point[i, 0] + (x_length_bot/10)*1.5)
            x_13 = int(final_middle_point[i, 0] + (x_length_bot/10)*4)
            y_0 = int(final_middle_point[i, 1] - (y_length - final_middle_point[i, 1])/2)
            y_1 = int(final_middle_point[i, 1] + (y_length - final_middle_point[i, 1])/2)
        factor_0 = 630
        factor_1 = 570
        for i in range (14):
            tmp = np.array([[factor_0, x_00, y_0], [factor_0, x_01, y_0], [factor_0, x_02, y_0], [factor_0, x_03, y_0],
                                    [factor_1, x_10, y_1], [factor_1, x_11, y_1], [factor_1, x_12, y_1], [factor_1, x_13, y_1]])
            auto_ini_pos.append(tmp)
        print auto_ini_pos[0][0]
        print auto_ini_pos[0]

    img = cv2.imread('Data/Radiographs/01.tif', 0)
    init_guess_img = img.copy()
    cropped_img = crop(init_guess_img, 1000, 500, 1000, 1000)
    img_contains_each_incisor = crop(img, 1000, 500, 1000, 1000)
    img_contains_each_incisor = median_filter(img_contains_each_incisor)
    img_contains_each_incisor = bilateral_filter(img_contains_each_incisor)
    img_contains_each_incisor = sobel(img_contains_each_incisor)
    mask_correct_lm = np.zeros((1000, 1000), dtype='uint8')
    mask_target_lm = np.zeros((1000, 1000), dtype='uint8')
    for i in range(8):
        Nr_incisor = i + 1
        source = 'Data\Landmarks\c_landmarks\landmarks9-%d.txt' % Nr_incisor
        lm = Landmarks(source).show_points()

        """Initial position guess"""

        ini_pos = auto_ini_pos[0]
        s = ini_pos[i, 0]
        t = [ini_pos[i, 1], ini_pos[i, 2]]
        Golden_lm = load_training_data(Nr_incisor)
        Golden_lm = rescale_withoutangle(gpa(Golden_lm)[2], t, s )# Y
        cropped_img = cropped_img.copy()
        img = median_filter(cropped_img)
        img = bilateral_filter(img)
        # print raw_img
        #img = top_hat_transform(img)
        #img = bottom_hat_transform(img)
        img = sobel(img)

        """Drawing initial guess"""
        #cv2.imshow('first golden model', drawlines(crop_img_init_guess, Golden_lm))
        img_contains_each_incisor = drawlines(img_contains_each_incisor, Golden_lm)
        cv2.imwrite('Data\Configure_with_auto_init\_auto_init_guess_incisor_img9-%d.tif' % Nr_incisor, drawlines(img_contains_each_incisor, Golden_lm))
        #cv2.imwrite('Data\Configure\correct_lm_incisor-%d.tif' % Nr_incisor, drawlines(crop_correct_lm, lm))
        #cv2.waitKey(0)


        X = active_shape_model(Golden_lm, img, max_iter = 10, Nr_incisor = Nr_incisor, search_length = 2)
      #  X = active_shape_model(Golden_lm, cropped_img, max_iter = 10, Nr_incisor = Nr_incisor, search_length = 2)
       # MSE = evaluation(X, lm)
       # print('The Distance SME of %d incisor is %3.4f' % (Nr_incisor, MSE))
       # F-measure
        Nr_incisor = i + 1
        source = 'Data\Landmarks\c_landmarks\landmarks1-%d.txt' % Nr_incisor
        lm = Landmarks(source).show_points()
        #   print(lm)
        for j in range(np.shape(lm)[0]):
            lm[j, 0] = int(lm[j, 0] - 1000)
            lm[j, 1] = int(lm[j, 1] - 500)
        #  print lm[i, 0]
        #   print lm[i, 1]
        for j in range(np.shape(lm)[0]):
            mask_correct_lm[int(lm[j, 1]), int(lm[j, 0])] = 1

        #  print lm[i, 0]
        #   print lm[i, 1]
        for j in range(np.shape(lm)[0]):
            mask_target_lm[int(lm[j, 1]), int(lm[j, 0])] = 2

        for j in range(np.shape(lm)[0]-1):
            mask_correct_lm = cv2.line(mask_correct_lm, (int(lm[j, 0]), int(lm[j, 1])),
                     (int(lm[(j + 1), 0]), int(lm[(j + 1), 1])),
                     (255, 255, 0), 2)
            mask_correct_lm = cv2.line(mask_correct_lm, (int(lm[0, 0]), int(lm[0, 1])),
                     (int(lm[39, 0]), int(lm[39, 1])),
                     (255, 255, 0), 2)
        drawlines(mask_correct_lm, lm)
        drawlines(mask_target_lm, X)

       # cv2.imshow('lm', mask_correct_lm)
        #cv2.imshow('X', mask_target_lm)
      #  cv2.waitKey(0)


       #for i r


    cv2.imwrite('Data\configure_to_Fmeasure\F_measure_lm.tif',mask_correct_lm )
    cv2.imwrite('Data\configure_to_Fmeasure\F_measure_X.tif',mask_target_lm )


