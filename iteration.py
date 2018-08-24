
import pre_process
from timeit import Timer
import numpy as np
import math
from Plotter import drawlines
import cv2
import PCA
from PCA import ASM
from pre_process import Landmarks
from pre_process import rescale_withoutangle
from pre_process import load, as_vectors
from Figure_denoise import crop
from GPA import gpa, align_params
from util import load_training_data
from pre_process import find_the_normal_to_lm
from Figure_denoise import median_filter, sobel, bilateral_filter, median_filter, canny
from pre_process import Landmarks
from Figure_denoise import load_image

import Figure_denoise
import Plotter

def active_shape_model(X, testimg, max_iter, Nr_incisor,normal_search_range):
    """

    :param X:   Init Guess
    :param testimg:   target image
    :param max_iter:   total iteration limitation
    :param Nr_incisor:   which nr. of incisor is used to do the asm
    :return:     a model describe the target incisor on the image
    """
    img = bilateral_filter(testimg)
    img = median_filter(img)
    img = sobel(img)
    X = Landmarks(X).show_points()
    # Initial value
    nb_iter = 0
    n_close = 0
    total_s = 1
    total_theta = 0
    # Begin to iterate.
    lm_objects = load_training_data(Nr_incisor)
    landmarks_pca = PCA.ASM(lm_objects)
    while (n_close < 16 and nb_iter <= max_iter):


        # 1. Examine a region of the image around each point Xi to find the
        # best nearby match for the point
        """
        Y, n_close, quality = self.__findfits(X, img, gimg, glms, m)
        if quality < best:
               best = quality
            best_Y = Y
        Plotter.plot_landmarks_on_image([X, Y], testimg, wait=False,
                                           title="Fitting incisor nr. %d" % (self.incisor_nr,))

        # no good fit found => go back to best one
        if nb_iter == max_iter:
            Y = best_Y
        """
        Y =  get_max_along_normal(X, normal_search_range, img)
        #print Y
        # 2. Update the parameters (Xt, Yt, s, theta, b) to best fit the
        # new found points X
        b, t, s, theta = parameter_update(X, Y, Nr_incisor)
        """ 
        Apply constraints to the parameters, b, to ensure plausible shapes
        We clip each element b_i of b to b_max*sqrt(l_i) where l_i is the
        corresponding eigenvalue.
        """
        b = np.clip(b, -3, 3)
        # t = np.clip(t, -5, 5)
        # limit scaling
        s = np.clip(s, 0.95, 1.05)
        if total_s * s > 1.20 or total_s * s < 0.8:
            s = 1
        total_s *= s
        # limit rotation
        theta = np.clip(theta, -math.pi / 8, math.pi / 8)
        if total_theta + theta > math.pi / 4 or total_theta + theta < - math.pi / 4:
            theta = 0
        total_theta += theta
        """Finish limitation"""

        # The positions of the model points in the image, X, are then given
        # by X = TXt,Yt,s,theta(X + Pb)
        """By updating X and apply equation 4, to map the dataset into image coordinates"""
        X = Landmarks(X).as_vector()
        X = Landmarks(X + np.dot(landmarks_pca.pc_modes, b)).T(t, s, theta)
        #Plotter.plot_landmarks_on_image([X_prev, X], testimg, wait=False,
        #                                    title="Fitting incisor nr. %d" % (Nr_incisor,))
        X = X.show_points()
        img2 = img.copy()
        """Calibration"""
        final_image = drawlines(img2, X)
        cv2.imshow('iteration results ', final_image)
        cv2.waitKey(10)
        nb_iter += 1
        #print('this is the %d iteration'% nb_iter)

    cv2.imwrite('Data\Configure\incisor-%d.tif' % Nr_incisor, final_image)
    return X





def parameter_update(X, Y, Nr_incisor):
    """This parts strictly follow  Tim Cootes's paper as Protocol 1
    Y should be given pointset or initial guess
    X initial guess"""
    lm_objects = load_training_data(Nr_incisor)
    landmarks_pca =  PCA.ASM(lm_objects)
    b = np.zeros(landmarks_pca.pc_modes.shape[1])
    b_prev = np.ones(landmarks_pca.pc_modes.shape[1])
    i = 0
    X = Landmarks(X).as_vector()
    Y = Landmarks(Y)
    while (np.mean(np.abs(b - b_prev)) >= 1e-14):
        i += 1
        # 2. Generate the model point positions using x = X + Pb
        x = Landmarks(X + np.dot(landmarks_pca.pc_modes, b))

        # 3. Find the pose parameters (Xt, Yt, s, theta) which best align the
        # model points x to the current found points Y
        t, s, theta = align_params(x, Y)
        #t, s, theta = align_params(x.get_crown(is_upper), Y.get_crown(is_upper))

        # 4. Project Y into the model co-ordinate frame by inverting the
        # transformation T
        y = Y.invT(t, s, theta)

        # 5. Project y into the tangent plane to X by scaling:
        # y' = y / (y*X).
        yacc = Landmarks(y.as_vector() / np.dot(y.as_vector(), X.T))

        # 6. Update the model parameters to match to y': b = PT(y' - X)
        b_prev = b

        b = np.dot(landmarks_pca.pc_modes.T, (yacc.as_vector() - X))

        # 7. If not converged, return to step 2
    """
    The discription is discussed in the paper.....
    This function describe the opearation of changing from given lm to the model with convergended b.
    We should iterate b first to make it convergence, then we get the final value of b, which is the only
    meaningful value in this section.  The eigen value b refers to a point in high dimension space which also 
    represents a model.
    
    In this case, if another given 'model', like initial guess or lm, we must find this b. The way to find this b,
    is to used this iteration method,
    
    After b is convergence, then we know b is static and hence the model is steady, we then abstract the 
    pose(xt, yt, theta, s) for active shape model. 
    
    
    Generally speaking, accoording to equation 2 and 3, If b maintains the same, then x = xbar, then two models alignes well.
    
    For some practical information. 
    Feed Xbar (PCA MEAN) and Y (Final goal state model).
    By iteration using eqaition 2, we have new y (which is also the x in equation).
    then we align the new obtained y to Final gaol state Y. and we have a tmp parameters.
    
    The reason why we cannot get the final parameters after the alignment should be supported by the appendix 6 in paper.
    
    In this case, we update b by using equation 3 and check if it is convergent.
    If not, it shows the the previous obtained value y is not like the given model Y.
    Then we need to continue iteration until b convergent.
    What we want to get from this function is the transmission pose parameters.
    """
    return b, t, s, theta

def get_max_along_normal(lm, length, img):
    """lm: is the initial guess
       length: the search range of single points
       img: should be single channel grey level picture
       return: The maximum value set along each normal"""
    normal = find_the_normal_to_lm(lm)
    max_points_along_normal = np.zeros((len(lm), 2))
    for i in range(len(lm)):
        count = 0
        intensity = np.zeros(length)
        for j in range(-length, length-1): # for comparing intensity[j] > intensity[j+1]:
            if abs(int(lm[i,0]+ j + 1)) >= 999 or abs(int(lm[i,1] + normal[i]*(j+1))) >= 999:
                intensity[j + 1] = intensity[j]
                count += 1
            else:
                intensity[j+1] = img[int(lm[i,0]+ j + 1),  int(lm[i,1] + normal[i]*(j+1))] # the multiple part will be too big.
            if abs(intensity[j+1] - intensity[j]) > abs(intensity[j] - intensity[j]):
                max_points_along_normal[i, :] = [lm[i,0] + j + 1,  lm[i,1] + normal[i]*(j+1)]
        #print(count) # find how many approximation
    return max_points_along_normal

def grey_level_model(lm, index, length):
    """
       lm: is the initial guess
       length: the search range of single points
       index: The point
       img: should be single channel grey level picture
       return: The maximum value set along each normal
       intensity[g01, g02, g03,...]
                [g11, g12, g13,...]
    """
    normal = find_the_normal_to_lm(lm)
    #gradient = np.zeros((2 * length - 1, 14))
    grey_model_all_points = []
    for i in range(len(lm)):
        gradient_set = np.zeros((14, 2 * length + 1))
        for img_nr in range(14-1):
            img = cv2.imread('Data/Radiographs/%02d.tif' % (img_nr+2), 0)
            for j in range(-length, length):  #
                #if abs(int(lm[i,0]+ j )) >= 999 or abs(int(lm[i, 1] + normal[i] * j)) >= 999:
                    # if j == length - 1:
                    #     raise('Intensity on the last point alonfg this noraml is too large')
                    #else:
                    #gradient_set[img_nr,j] = gradient_set[img_nr,j]
                #else:
                gradient_set[img_nr, j] = img[int(lm[index, 0] + j + 1), int(lm[index, 1] + normal[index] * (j + 1))] - \
                                              img[int(lm[index, 0] + j ), int(lm[index, 1] + normal[index] * j)]
                #print (img[int(lm[i, 0] + j + 1), int(lm[i, 1] + normal[i] * (j + 1))])
           # print(gradient_set[img_nr, :])
            gradient_set[img_nr, :] = gradient_set[img_nr, :] / (sum(gradient_set[img_nr, :])+1) # plus 1 for avoid zero dviator
    cov = np.cov(gradient_set) # this may have axis choosing issues.
    mean = np.mean(gradient_set, axis = 0)
    print(mean)
    print(np.shape(mean)) # the should be an 2length+1 array
    return cov, mean



def find_the_best_score(lm, testimg, search_length, glm_range, cov, mean):
    target = np.zeros(glm_range)
    normal = find_the_normal_to_lm(lm)
    target = np.zeros(glm_range)
    score = np.zeros(search_length - glm_range)
    max_score =np.zeros(len(lm))
    for i in range(lm):
        for k in range(search_length - glm_range):
            for j in range(-search_length, -search_length+glm_range):
                target[j] = testimg[int(lm[search_length, 0] + j ), int(lm[search_length, 1] + normal[search_length] * j)]
            score[k]= (target - mean).T \
                    .dot(cov) \
                    .dot(target - mean)

        if score[k] >= score[k+1]:
            max_score[i] = score[k]
    return max_score








def evaluation(X, Golden_lm):
    """

    :param X: The ASM model lm of the image
    :param Golden_lm: The correct lm of image
    :return: the square mean error
    """
    error_single = 0
    for i in range(np.shape(X)[0]):
        error_single += (X[i, 0] - Golden_lm[i, 0]) + (X[i, 1] - Golden_lm[i, 1])**2
    return error_single

if __name__ == '__main__':
    glm_range = 5
    max_iter = 10
    source = 'Data\Landmarks\c_landmarks\landmarks1-%d.txt' % 1
    lm = Landmarks(source).show_points()
    print (grey_level_model(lm, 1,  glm_range))

    # for i in range(8):
    #     Nr_incisor = i + 1
    #     source = 'Data\Landmarks\c_landmarks\landmarks1-%d.txt' % Nr_incisor
    #     lm = Landmarks(source).show_points()
    #    # print lm # you can print it to check differences.
    #
    #     """Initial position guess"""
    #     ini_pos = np.array([[570, 360, 390], [620, 470, 390], [640, 570, 370], [640, 670, 370], [640, 400, 670],
    #                        [640, 490, 660], [620, 570, 670], [640, 650, 660]])
    #     s = ini_pos[i, 0]
    #     t = [ini_pos[i, 1], ini_pos[i, 2]]
    #     Golden_lm = load_training_data(Nr_incisor)
    #     Golden_lm = rescale_withoutangle(gpa(Golden_lm)[2], t, s )# Y
    #     img = cv2.imread('Data/Radiographs/01.tif', 0)
    #     init_guess_img = img.copy()
    #     crop_img_ori = crop(img, 1000, 500, 1000, 1000)
    #     crop_img_init_guess = crop(init_guess_img, 1000, 500, 1000, 1000)
    #     crop_correct_lm = crop(init_guess_img, 1000, 500, 1000, 1000)
    #     """Drawing initial guess"""
    #     #cv2.imshow('first golden model', drawlines(crop_img_init_guess, Golden_lm))
    #     cv2.imwrite('Data\Configure\init_guess_incisor-%d.tif' % Nr_incisor, drawlines(crop_img_init_guess, Golden_lm))
    #     cv2.imwrite('Data\Configure\correct_lm_incisor-%d.tif' % Nr_incisor, drawlines(crop_correct_lm, lm))
    #     #cv2.waitKey(0)
    #
    #
    #     X = active_shape_model(Golden_lm, crop_img_ori, max_iter = max_iter, Nr_incisor = Nr_incisor, normal_search_range = normal_search_range)
    #     MSE = evaluation(X, lm)
    #     print('The Distance SME of %d incisor is %3.4f' % (Nr_incisor, MSE))









