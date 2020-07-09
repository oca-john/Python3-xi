#!/usr/bin/python3

# coding:utf-8

from PySide2 import QtWidgets
import sys


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = App()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
