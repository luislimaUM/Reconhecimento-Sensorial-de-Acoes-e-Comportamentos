import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog 

#import training dataset of vehicles/non-vehicles
import glob
car = glob.glob('OwnCollection/vehicles/Far/*.png')
no_car = glob.glob('OwnCollection/non-vehicles/Far/*.png')

#print(len(car))
#print(len(no_car))

car_hog_accum = []

for i in car:
    image_color = mpimg.imread(i)
    image_gray  = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    car_hog_feature, car_hog_img = hog(image_color[:,:,0], 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    car_hog_accum.append(car_hog_feature)

X_car = np.vstack(car_hog_accum).astype(np.float64)  
y_car = np.ones(len(X_car))

nocar_hog_accum = []

for i in no_car:
    image_color = mpimg.imread(i)
    image_gray  = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    nocar_hog_feature, car_hog_img = hog(image_color[:,:,0], 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    nocar_hog_accum.append(nocar_hog_feature)

X_nocar = np.vstack(nocar_hog_accum).astype(np.float64)  
y_nocar = np.zeros(len(X_nocar))

X = np.vstack((X_car, X_nocar))
y = np.hstack((y_car, y_nocar))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
svc_model = LinearSVC()
svc_model.fit(X_train,y_train)

y_predict = svc_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True, fmt="d")

#print(cm)
print(classification_report(y_test, y_predict))

#comparaçao saidas
#Model_prediction = svc_model.predict(X_test[0:50])
#print(Model_prediction)
#Model_TrueLabel = y_test[0:50]
#print(Model_TrueLabel)

#improve the model
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,grid_predictions))

cap = cv2.VideoCapture(0)

def Detection(image):

    h_start = 0
    h_stop = 480
    masked_region = image[h_start:h_stop,:,:]
    resizing_factor = 1
    masked_region_shape = masked_region.shape
    L = masked_region_shape[1]/resizing_factor
    W = masked_region_shape[0]/resizing_factor
    masked_region_resized = cv2.resize(masked_region, (np.int(L), np.int(W)))
    masked_region_resized_R = masked_region_resized[:,:,0]

    pixels_in_cell = 16
    HOG_orientations = 11
    cells_in_block = 2
    cells_in_step = 3 

    masked_region_hog_feature_all, hog_img = hog(masked_region_resized_R,
                                     orientations = 11, 
                                     pixels_per_cell = (16, 16), 
                                     cells_per_block = (2, 2), 
                                     transform_sqrt = False, 
                                     visualize = True, 
                                     feature_vector = False)
    
    n_blocks_x = (masked_region_resized_R.shape[1] // pixels_in_cell)+1  
    n_blocks_y = (masked_region_resized_R.shape[0] // pixels_in_cell)+1

    #nfeat_per_block = orientations * cells_in_block **2 
    blocks_in_window = (64 // pixels_in_cell)-1 
    
    steps_x = (n_blocks_x - blocks_in_window) // cells_in_step
    steps_y = (n_blocks_y - blocks_in_window) // cells_in_step

    rectangles_found = []

    for xb in range(steps_x):
        for yb in range(steps_y):
            y_position = yb*cells_in_step
            x_position = xb*cells_in_step
            
            hog_feat_sample = masked_region_hog_feature_all[y_position : y_position + blocks_in_window, x_position : x_position + blocks_in_window].ravel()
            x_left = x_position * pixels_in_cell
            y_top = y_position * pixels_in_cell
            #print(hog_feat_sample.shape)  
        
            # predict using trained SVM
            test_prediction = svc_model.predict(hog_feat_sample.reshape(1,-1))
            # test_prediction = grid.predict(hog_feat_sample.reshape(1,-1))
        
            if test_prediction == 1: 
                rectangle_x_left = np.int(x_left * resizing_factor)
                rectangle_y_top = np.int(y_top * resizing_factor)
                window_dim = np.int(64 * resizing_factor)
                rectangles_found.append(((rectangle_x_left, rectangle_y_top + h_start),(rectangle_x_left + window_dim, rectangle_y_top + window_dim + h_start)))
            
    print(rectangles_found)

    Image_with_Rectangles_Drawn = np.copy(image)
    
    for rectangle in rectangles_found:
        cv2.rectangle(Image_with_Rectangles_Drawn, rectangle[0], rectangle[1], (0, 255, 0), 20)

    return Image_with_Rectangles_Drawn

while True:
    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
    cv2.imshow('Webcam', Detection(frame))
    if cv2.waitKey(1) == 13: #13 Enter Key
        break
        
cap.release() # camera release 
cv2.destroyAllWindows() 
