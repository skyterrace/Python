import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print face_cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
capture =  cv2.VideoCapture(0)
frec = open ("test.txt",'w+')
if capture.isOpened():
    #image = cv.QueryFrame(capture)
    [retval,img] = capture.read()
    #img_show = cv.CloneImage(image)
    cv2.imshow("pre",img)
    key=cv2.waitKey(1)
    while key != 27:
        [retval,img] = capture.read()
        #img = cv2.imread('face.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_area = 0
        for (x,y,w,h) in faces:
            if w*h > face_area:
                face_area = w*h
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        cv2.imshow("pre",img)
        key = cv2.waitKey(100)
        print face_area
        frec.writelines([str(face_area),'\n'])
#cv.ReleaseImage(img_show)
#cv.ReleaseCapture(capture)
capture.release()
frec.close()
cv2.destroyAllWindows()
