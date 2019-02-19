import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def LiveCamEdgeDetection(image_color):

    # threshold1 – first threshold for the hysteresis procedure.
    # threshold2 – second threshold for the hysteresis procedure.
    # Um limite muito alto pode perder informações importantes. Por outro lado, um limiar definido como muito baixo irá identificar falsamente informações irrelevantes (como ruído) como importantes. 

    threshold_1 = 30
    threshold_2 = 80
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)
    canny = cv2.Canny(image_blurred, threshold_1, threshold_2)
    #cornerHarris(img,block size, ksize, k)
    #corners = cv2.cornerHarris(canny, 5, 3, 0.1)
    #corners_dilate = cv2.dilate(corners, np.ones((8,8),np.uint8), iterations = 1)
    #image_color[corners > 0.01 * corners.max()] = [0,0,255]
    
    return canny

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
    cv2.imshow('Webcam', LiveCamEdgeDetection(frame))
    cv2.imshow('Webcam Video', frame)
    if cv2.waitKey(1) == 13: #13 Enter Key
        break
        
cap.release() # camera release 
cv2.destroyAllWindows() 