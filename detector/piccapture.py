import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray

camera = PiCamera()
camera.resolution = (800,600)
rawCapture = PiRGBArray(camera,size=(800,600))
path_pre = "./svmTrain/"

for rawFrame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
	image = rawFrame.array
    cv2.imshow("pre",image)
    key=cv2.waitKey(1) & 0xFF
    img_time=time.localtime()
    img_name = time.strftime("%Y%m%d%H%M%S.jpg",img_time)
	
	if key == ord('q'):
		break
    elif key>47 and key <58:
        img_name= path_pre+"{0}/".format(key-48)+img_name
        cv2.imwrite(img_name,image)
        print "save to ",img_name
    elif key >96 and key < 103:
        img_name= path_pre+"{0}/".format(key-87)+img_name
        cv2.imwrite(img_name,image)
        print "save to ",img_name
	rawCapture.truncate(0)
print("Quit")

capture.release()
cv2.destroyAllWindows()
