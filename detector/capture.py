import cv2
import time
capture =  cv2.VideoCapture(0)
path_pre = "./svmTrain/"
if capture.isOpened():
    #image = cv.QueryFrame(capture)
    [retval,image] = capture.read()
    #img_show = cv.CloneImage(image)
    cv2.imshow("pre",image)
    key=cv2.waitKey(1) & 0xFF
    while key != ord('q'):
        img_time=time.localtime()
        img_name = time.strftime("%Y%m%d%H%M%S.jpg",img_time)
        if key>47 and key <58:
            img_name= path_pre+"{0}/".format(key-48)+img_name
            cv2.imwrite(img_name,image)
            print "save to ",img_name
        elif key >96 and key < 103:
            img_name= path_pre+"{0}/".format(key-87)+img_name
            cv2.imwrite(img_name,image)
            print "save to ",img_name
        [retval,image] = capture.read()
        cv2.imshow("pre",image)
        key = cv2.waitKey(1) & 0xFF
print("Quit")

capture.release()
cv2.destroyAllWindows()
