import os
import numpy as np
import tensorflow as tf
import PIL
import cv2
from PIL import ImageOps, Image
import time
import pandas as pd
import imutils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import atan2,degrees



class PathSeg(object):
    
    # you can use the index of any category from the list for your desired segmentation
    #labels_list=['others', 'wall', 'building;edifice', 'sky', 'floor;flooring', 'tree', 'ceiling', 'road;route', 'bed', 'windowpane;window', 'grass', 'cabinet', 'sidewalk;pavement', 'person;individual;someone;somebody;mortal;soul', 'earth;ground', 'door;double;door', 'table', 'mountain;mount', 'plant;flora;plant;life', 'curtain;drape;drapery;mantle;pall', 'chair', 'car;auto;automobile;machine;motorcar', 'water', 'painting;picture', 'sofa;couch;lounge', 'shelf', 'house', 'sea', 'mirror', 'rug;carpet;carpeting', 'field', 'armchair', 'seat', 'fence;fencing', 'desk', 'rock;stone', 'wardrobe;closet;press', 'lamp', 'bathtub;bathing;tub;bath;tub', 'railing;rail', 'cushion', 'base;pedestal;stand', 'box', 'column;pillar', 'signboard;sign', 'chest;of;drawers;chest;bureau;dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace;hearth;open;fireplace', 'refrigerator;icebox', 'grandstand;covered;stand', 'path', 'stairs;steps', 'runway', 'case;display;case;showcase;vitrine', 'pool;table;billiard;table;snooker;table', 'pillow', 'screen;door;screen', 'stairway;staircase', 'river', 'bridge;span', 'bookcase', 'blind;screen', 'coffee;table;cocktail;table', 'toilet;can;commode;crapper;pot;potty;stool;throne', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove;kitchen;stove;range;kitchen;range;cooking;stove', 'palm;palm;tree', 'kitchen;island', 'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system', 'swivel;chair', 'boat', 'bar', 'arcade;machine', 'hovel;hut;hutch;shack;shanty', 'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle', 'towel', 'light;light;source', 'truck;motortruck', 'tower', 'chandelier;pendant;pendent', 'awning;sunshade;sunblind', 'streetlight;street;lamp', 'booth;cubicle;stall;kiosk', 'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box', 'airplane;aeroplane;plane', 'dirt;track', 'apparel;wearing;apparel;dress;clothes', 'pole', 'land;ground;soil', 'bannister;banister;balustrade;balusters;handrail', 'escalator;moving;staircase;moving;stairway', 'ottoman;pouf;pouffe;puff;hassock', 'bottle', 'buffet;counter;sideboard', 'poster;posting;placard;notice;bill;card', 'stage', 'van', 'ship', 'fountain', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter', 'canopy', 'washer;automatic;washer;washing;machine', 'plaything;toy', 'swimming;pool;swimming;bath;natatorium', 'stool', 'barrel;cask', 'basket;handbasket', 'waterfall;falls', 'tent;collapsible;shelter', 'bag', 'minibike;motorbike', 'cradle', 'oven', 'ball', 'food;solid;food', 'step;stair', 'tank;storage;tank', 'trade;name;brand;name;brand;marque', 'microwave;microwave;oven', 'pot;flowerpot', 'animal;animate;being;beast;brute;creature;fauna', 'bicycle;bike;wheel;cycle', 'lake', 'dishwasher;dish;washer;dishwashing;machine', 'screen;silver;screen;projection;screen', 'blanket;cover', 'sculpture', 'hood;exhaust;hood', 'sconce', 'vase', 'traffic;light;traffic;signal;stoplight', 'tray', 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin', 'fan', 'pier;wharf;wharfage;dock', 'crt;screen', 'plate', 'monitor;monitoring;device', 'bulletin;board;notice;board', 'shower', 'radiator', 'glass;drinking;glass', 'clock', 'flag']
    
    
    def __init__(self):
        self.interpreter = None
        self.input_shape = None
        self.input_details = None
        self.output_details = None
        self.load()
        self.ongoing_predict=True
        
        
    
    def load(self):
                
        print("loading the model...")
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="lite-model_deeplabv3-mobilenetv2-ade20k_1_default_2.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get Input shape
        self.input_shape = self.input_details[0]['shape']
        print("ok done!")
        
    def img_enhancement(self,img_ycrcb):  
        
        # apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit = 10, tileGridSize=(10,10))# clipLimit is the threshold for contrast limiting 
        img_ycrcb[:,:,0] = clahe.apply(img_ycrcb[:,:,0]) + 30
        
        hist_eq = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        pil_image = Image.fromarray(hist_eq) # opencv image to PIL format
        return pil_image
        
    def fillholes(self,img):
        drawing2 = np.zeros_like(img, np.uint8)
        contours= cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea) 
        cv2.drawContours(drawing2, [c], 0, (255, 255, 255), -1)
        
        return cv2.dilate(drawing2, None, iterations=78)
        
    def AngleBtw2Points(self,pointA, pointB):
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
        return degrees(atan2(changeInY,changeInX)) #remove degrees if you want your answer in radians

    def find_lane_pixels(self,binary_warped):
        # most of this block is used from this gitHub https://github.com/xueyizou/CarND-Advanced-Lane-Lines-Finder
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0]//2)
        
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 8
        # Set the width of the windows +/- margin
        margin = 500
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        rightx_current = rightx_base

        # Create empty lists to receive center lane pixel indices
        right_lane_inds = []
        

        # Step through the windows one by one
        for i,window in enumerate(range(nwindows)):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            
            # Identify the nonzero pixels in x and y within the window #
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            
            
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
            

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        return rightx, righty 

    def fit_polynomial(self,binary_warped):
    
        h, w= binary_warped.shape[:2]
        ref_frame_pt= (int(w/2),h)
        # Find our lane pixels first
        rightx, righty = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        right_fit = np.polyfit(righty, rightx, 2)
        
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
        try:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            
        except TypeError:
            # Avoids an error if `right_fit` is still none or incorrect
            print('The function failed to fit a line!')
            right_fitx = 1*ploty**2 + 1*ploty

        
        verts=np.array(list(zip(right_fitx.astype(int),ploty.astype(int))))
        
        ref_poly_pt = verts[int(len(verts)/3)]
        curve=abs(self.AngleBtw2Points(ref_poly_pt,ref_frame_pt)) - 90
        ##print(curve)
        return curve
    
    def run(self, img_arr): # the run gets the input automatically from where you specified in the testmangae.py
        if img_arr is None:
            print("frame is None")
        try:
            if (self.ongoing_predict):
                self.ongoing_predict=False
                img_ycrcb = cv2.cvtColor(img_arr,cv2.COLOR_BGR2YCrCb)
                img_arr= self.img_enhancement(img_ycrcb)
                print("image enhancement done!")
                steering,throtl= self.inference(img_arr)
                print("timedelay = ",timedelay)
                if (abs(steering) < 0.3):
                    return steering,throtl
                if (abs(steering) > 0.3):
                    return steering,0.25                
                    
            if (self.ongoing_predict==False):
                self.ongoing_predict=True
                time.sleep(0.6)
                return 0.0,0.0
                
            
        except:        
            print("No frame available")
            return 0.0, 0.0 # uncomment this at the end

    def shutdown(self):
        
        cv2.destroyAllWindows()
        pass
    
    def inference(self, img_arr):
        #img_arr.show()
        print("inferring the image")
        start = time.time()
        img_arr= img_arr.resize((513,513))

        np_new_img = np.array(img_arr)
        np_new_img = (np_new_img/255).astype('float32')

        np_new_img = np.expand_dims(np_new_img, 0)
        
        input_data = np_new_img
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        
        print("making prediction")  
        raw_prediction = self.interpreter.tensor(self.interpreter.get_output_details()[0]['index'])()
        #print("raw_prediction")
        seg_map = tf.argmax(tf.image.resize(raw_prediction, (640,480)), axis=3)
        seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)        
        #print("seg_map",seg_map)    
        
        outputbitmap = np.zeros((480,640)).astype(int)
        
        result=np.where(seg_map == 4,255,0)
        result = Image.fromarray(np.uint8(result)).convert('L') 
             
        cv_image=np.array(result)   
        
        
        filled_holes=self.fillholes(cv_image)
        
        steering_angle= self.fit_polynomial(filled_holes)
        
        steering = float(steering_angle)
        
        steering=np.interp(steering,[-15,15],[-1,1]) # mapping the degrees calculated to -1 and 1
        
        
        if (steering >0):
            print ("to the right")
        else:
            print("to the left")
            
        throttle = 0.2
        
        print("steering_angle",steering)
        end = time.time()
        print("it took:" + str(end - start))
        return steering, throttle

    















