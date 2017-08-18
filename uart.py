import serial
import time
send_buf = bytearray(6)
recv_buf = bytearray(6)
ser = serial.Serial("/dev/ttyAMA0",baudrate=9600)
def main():
#	ser.open()
	i=0
	ser.flushInput()
#	print ser.portstr
	send_buf[0]=30
	send_buf[1]=31
	send_buf[2]=32
	send_buf[3]=33
	send_buf[4]=34
	send_buf[5]=35
	ser.write(send_buf)
	time.sleep(0.5)
	while True:
		count = ser.inWaiting()
		for i in range(0,count):
			if i > 5 :
				print "last",ser.read()
			else:
				recv_buf[i]=ser.read()
#			time.sleep(0.001)
#			recv=ser.read()
#			ser.write(recv)
		print "get",count,"bytes",recv_buf[0],recv_buf[1]
#		ser.flushInput()
		print "run..."
		time.sleep(0.5)
		recv_buf[1] = recv_buf[1]+1
		sc = ser.write(recv_buf)
		print "send",sc,"bytes",recv_buf[0],recv_buf[1]
		time.sleep(0.5)
if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		if ser!= None:
			ser.close()

