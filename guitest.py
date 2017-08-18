import turtle
import sys
from PySide.QtCore import *
from PySide.QtGui import *
class TurtleControl(QWidget):
    def __init__(self,turtle):
        super(TurtleControl,self).__init__()
        self.turtle = turtle

        self.left_btn=QPushButton("Left",self)

        self.controlsLayout = QGridLayout()
        self.controlsLayout.addWidget(self.left_btn,0,0)
        self.setLayout(self.controlsLayout)

##window = turtle.Screen()
##babbage = turtle.Turtle()

app = QApplication(sys.argv)
##control_window = TurtleControl(babbage)
##control_window.show()

label = QLabel("Hello World")
label.show()

app.exec_()
sys.exit()
