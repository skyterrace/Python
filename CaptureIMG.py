from cv2 import *
capture =  VideoCapture(0)
if capture.isOpened():
    #image = cv.QueryFrame(capture)
    [retval,image] = capture.read()
    #img_show = cv.CloneImage(image)
    cv.ShowImage("pre",cv.fromarray(image))
    key=cv.WaitKey(1)
    while key != 27:
        [retval,image] = capture.read()
        cv.ShowImage("pre",cv.fromarray(image))
        key = cv.WaitKey(1)
print("quit")
#cv.ReleaseImage(img_show)
#cv.ReleaseCapture(capture)
capture.release()
