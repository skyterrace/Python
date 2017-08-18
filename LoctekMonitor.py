import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

send_buf = bytearray(20)
recv_buf = bytearray(20)
ser = serial.Serial("COM3",baudrate=115200)

x = np.zeros(100)
obj_x = np.zeros(100)

fig = plt.figure()
ax_light = fig.add_subplot(211)
ax_light.set_ylim(0,60)
ax_light.set_title("Light")
ax_light.set_ylabel("Light")
line_light, = ax_light.plot(x)

ax_obj = fig.add_subplot(212)
ax_obj.set_ylim(0,50)
ax_obj.set_title("Obj")
ax_obj.set_ylabel("Obj")
ax_obj.hold(None)
line_obj, = ax_obj.plot(obj_x)

def update_light(data):  
    line_light.set_ydata(data)  
    return line_light,
def data_gen_light():
    while True:  
        count = ser.inWaiting()
        for i in range(0,count):
                if i > 19 :
                        print "warning! more than 20 bytes, last",ser.read()
                else:
                        recv_buf[i]=ser.read()
        if count == 20:
                frame_ch2o = recv_buf[3]*256+recv_buf[4]
                frame_objtemp = recv_buf[5]*256+recv_buf[6]
                frame_ambtemp = recv_buf[7]*256+recv_buf[8]
                frame_temp = recv_buf[9]*256+recv_buf[10]
                frame_humidity = recv_buf[11]*256+recv_buf[12]
                frame_light = recv_buf[13]*256+recv_buf[14]
                frame_airgrade = recv_buf[15]/16

                print "CH2O=",frame_ch2o,", OBJ=",frame_objtemp,", AMB=",frame_ambtemp,", TEMP=",frame_temp,", HMD=",frame_humidity,", LIGHT=",frame_light/100.0,", AIR=",frame_airgrade
                #x = x[1:]+frame_light/100.0
                i=0
                while(i<99):
                        x[i]=x[i+1]
                        obj_x[i]=obj_x[i+1]
                        i+=1
                x[99] = frame_light/100.0
                obj_x[99] = frame_objtemp/100.0
                
                #ax_obj.clear()
                ax_obj.plot(obj_x)
                
        elif count > 0:
                print "error... the frame length is not 20 bytes (", count,")"
        time.sleep(0.1)
        yield x
def main():
#       ser.open()
        i=0
        ser.flushInput()
#       print ser.portstr
        send_buf[0]=30
        send_buf[1]=31
        send_buf[2]=32
        send_buf[3]=33
        send_buf[4]=34
        send_buf[5]=35
#       ser.write(send_buf)
        time.sleep(0.5)

        t = np.arange(0,100,1)


        ani = animation.FuncAnimation(fig, update_light, data_gen_light, interval=100)  
        plt.show()  
if __name__ == '__main__':
        try:
                main()
        except KeyboardInterrupt:
                if ser!= None:
                        ser.close()


