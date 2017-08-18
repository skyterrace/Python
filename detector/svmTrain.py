import numpy as np
import cv2
import os
import time

def cutoutImage(image):
        height, width, channel = image.shape
        centerX = width/2
        centerY = height/2
        marginX = width*0.25
        marginY = height*0.3
        startX = np.int(centerX - marginX)
        endX = np.int(centerX + marginX)
        startY = np.int(centerY - marginY)
        endY = np.int(centerY + marginY)
        image_cutout = image[startY:endY, startX:endX, :]
        return image_cutout
def getObjectMask(image):
        SR_WIDTH = 64
        width = image.shape[0]
        height = image.shape[1]

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_sr = cv2.resize(image_gray, (SR_WIDTH, SR_WIDTH*width/height))

        image_dft = cv2.dft(np.float32(image_sr), flags = cv2.DFT_COMPLEX_OUTPUT)
        image_dft_mag = np.sqrt(image_dft[:,:,0]**2+image_dft[:,:,1]**2)

        spectralResidual = np.exp(np.log(image_dft_mag) - cv2.boxFilter(np.log(image_dft_mag), -1, (3,3)))

        image_dft[:,:,0] = image_dft[:,:,0]*spectralResidual/image_dft_mag
        image_dft[:,:,1] = image_dft[:,:,1]*spectralResidual/image_dft_mag

        image_dft = cv2.dft(image_dft,flags = (cv2.DFT_INVERSE |cv2.DFT_SCALE))
        image_dft_mag = image_dft[:,:,0]**2+image_dft[:,:,1]**2
        cv2.normalize(cv2.GaussianBlur(image_dft_mag, (9,9), 3, 3), image_dft_mag, 0., 1., cv2.NORM_MINMAX)

        image_blurred = cv2.blur(image_dft_mag, (9,9))
        (_, image_thresh) = cv2.threshold(image_blurred, 0.15, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        image_closed = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)

        image_closed = cv2.erode(image_closed, None, iterations = 4)
        image_closed = cv2.dilate(image_closed, None, iterations = 4)

        image_mask = cv2.resize(image_closed, (height, width))
        (_, image_mask) = cv2.threshold(image_mask, 225, 255, cv2.THRESH_BINARY)
        image_mask = np.uint8(image_mask)
        return image_mask
def getObjectImage(image, object_mask):
        (cnts, _) = cv2.findContours(object_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = cv2.minAreaRect(c)
        startWidth = np.int(rect[0][1]-rect[1][1]/2)
        endWidth = np.int(rect[0][1]+rect[1][1]/2)
        startHeight = np.int(rect[0][0]-rect[1][0]/2)
        endHeight = np.int(rect[0][0]+rect[1][0]/2)
        image_sub = image[startWidth:endWidth, startHeight:endHeight]
        return image_sub
    
def findObjectInImage(image):
        image_cutout = cutoutImage(image)
        object_mask = getObjectMask( image_cutout )
        image_obj = getObjectImage(image_cutout, object_mask)
        return image_obj
    
def buildHOGDescriptor():
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        return hog
def reckonHOGFeature(hog, image):
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)
        hist = hog.compute(image,winStride,padding,locations)
        return hist

def getTrainSampleFiles(path_str):
        files = []
        resps = []
        for file in os.listdir(path_str):
                #resps.append(int(file))
                if os.path.isdir(path_str + file):
                        for subfile in os.listdir(path_str + file):
                                files.append(path_str + file + "/" + subfile)
                                resps.append(int(file))
        return files, resps
    
def genTrainSamples(path_str):
        image_files, hog_resp = getTrainSampleFiles(path_str)

        HOG_SIZE = (128, 128)
        hog = buildHOGDescriptor()
        hog_hist = []
        for file in image_files:
                image = cv2.imread(file)
                #image_obj = findObjectInImage(image)
                image_cutout = cutoutImage(image)
                object_mask = getObjectMask( image_cutout )
                image_obj = getObjectImage(image_cutout, object_mask)
                #image_obj = image_cutout
                image_hog = cv2.resize(image_obj, HOG_SIZE)
                hist = reckonHOGFeature(hog, image_hog)
                hog_hist.append( hist )

                print "Training ",file
                cv2.imshow("image_cutout",image_cutout)
                #cv2.imshow("image_mask",object_mask)
                cv2.imshow("image_obj",image_obj)
                cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return hog_hist, hog_resp

hogData, hogLabels = genTrainSamples("./svmTrain/")
trainData = np.array(hogData, dtype = np.float32)
trainLabels = np.array(hogLabels, dtype = np.int) 
svm = cv2.SVM()
svm_params = dict(kernel_type = cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, C=1, gamma=0.5)
#svm.setType(cv2.SVM_C_SVC)
#svm.setKernel(cv2.SVM_RBF)
#svm.setC(1) 
svm.train(trainData, trainLabels, params=svm_params)
svm.save("svmTrain_log_svm_model.yml");
print "Done!"
