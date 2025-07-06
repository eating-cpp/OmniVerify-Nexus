import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 左侧按钮布局
        left_layout = QVBoxLayout()
        button1 = QPushButton('主页')
        button2 = QPushButton('代理')
        button3 = QPushButton('配置选项')
        button4 = QPushButton('设置')
        button5 = QPushButton('日志')
        button6 = QPushButton('关于')

        button1.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        button2.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        button3.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        button4.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        button5.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        button6.clicked.connect(lambda: self.stack.setCurrentIndex(5))

        left_layout.addWidget(button1)
        left_layout.addWidget(button2)
        left_layout.addWidget(button3)
        left_layout.addWidget(button4)
        left_layout.addWidget(button5)
        left_layout.addWidget(button6)

        # 右侧堆叠窗口
        self.stack = QStackedWidget()
        page1 = QLabel('主页内容')
        page2 = QLabel('代理内容')
        page3 = QLabel('配置选项内容')
        page4 = QLabel('设置内容')
        page5 = QLabel('日志内容')
        page6 = QLabel('关于内容')

        self.stack.addWidget(page1)
        self.stack.addWidget(page2)
        self.stack.addWidget(page3)
        self.stack.addWidget(page4)
        self.stack.addWidget(page5)
        self.stack.addWidget(page6)

        # 整体布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.stack)

        self.setLayout(main_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
