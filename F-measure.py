import numpy as np
import cv2
from Figure_denoise import crop
from pre_process import Landmarks
if __name__ == '__main__':
    img_raw = cv2.imread('Data/Radiographs/01.tif', 0)
    print(np.shape(img_raw))
    crop_img_ori = crop(img_raw, 1000, 500, 1000, 1000) # (y, x, xx, yy)
    #cv2.imshow('cropped', crop_img_ori)
   # cv2.waitKey(0)
    mask = np.zeros((1000, 1000), dtype='uint8')


    for incisor in range(8):
        Nr_incisor = incisor + 1
        source = 'Data\Landmarks\c_landmarks\landmarks1-%d.txt' % Nr_incisor
        lm = Landmarks(source).show_points()
     #   print(lm)
        for i in range(np.shape(lm)[0]):
            lm[i, 0] = int(lm[i, 0] - 1000)
            lm[i, 1] = int(lm[i, 1] - 500)
          #  print lm[i, 0]
         #   print lm[i, 1]
        for i in range(np.shape(lm)[0]):
            mask[int(lm[i, 1]),int(lm[i, 0])] = 1

 #       for i in range(1000):# x
#            for j in range(1000): # y
#                if i < lm[]
        for i in range(np.shape(lm)[0]-1):
            mask = cv2.line(mask, (int(lm[i, 0]), int(lm[i, 1])),
                     (int(lm[(i + 1), 0]), int(lm[(i + 1), 1])),
                     (255, 255, 0), 2)
            mask = cv2.line(mask, (int(lm[0, 0]), int(lm[0, 1])),
                     (int(lm[39, 0]), int(lm[39, 1])),
                     (255, 255, 0), 2)
    cv2.imshow('lm', mask)
    cv2.waitKey(0)

