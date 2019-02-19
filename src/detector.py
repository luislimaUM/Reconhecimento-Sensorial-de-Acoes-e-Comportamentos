import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog # The Histogram of Oriented Gradient (HOG)
from skimage import exposure

def LiveCamEdgeDetection(image_color):

    # threshold1 – first threshold for the hysteresis procedure.
    # threshold2 – second threshold for the hysteresis procedure.
    # Um limite muito alto pode perder informações importantes. Por outro lado, um limiar definido como muito baixo irá identificar falsamente informações irrelevantes (como ruído) como importantes. 
    threshold_1 = 30
    threshold_2 = 80
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)
    canny = cv2.Canny(image_blurred, threshold_1, threshold_2)
    
    #detect corners
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(canny, None)
    print ("Number of keypoints Detected: ", len(keypoints))
    image = cv2.drawKeypoints(image_color, keypoints, None, color = (255,0,0))
    
    return image

def HogExtraction(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #x_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize = 7)
    features, hog_image = hog(image_gray, 
                                   orientations = 9, 
                                   pixels_per_cell = (16, 16), 
                                   cells_per_block = (1, 1), 
                                   transform_sqrt = False, 
                                   visualize = True, 
                                   feature_vector = False)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 2))

    return hog_image_rescaled


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
    #cv2.imshow('Webcam Video', frame)
    cv2.imshow('HOG Features',HogExtraction(frame))
    cv2.imshow('Webcam', LiveCamEdgeDetection(frame))
    if cv2.waitKey(1) == 13: #13 Enter Key
        break
        
cap.release() # camera release 
cv2.destroyAllWindows() 