import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QGridLayout)
from PyQt6.QtCore import Qt, QProcess, QUrl, QObject, pyqtSignal
from PyQt6.QtGui import QIcon
from pathlib import Path


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Détecteur de somnolence')

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.header = QLabel('Détecteur de somnolence',
                             alignment=Qt.AlignmentFlag.AlignCenter)
        self.header.setObjectName('header')

        self.subHeader = QLabel('Configuration des seuils',
                                alignment=Qt.AlignmentFlag.AlignCenter)
        self.subHeader.setObjectName('subHeader')
        self.standardChoice = QPushButton('Seuils fixes')

        self.personalChoice = QPushButton('Seuils adaptatifs')

        self.setWindowIcon(QIcon('single-wheel-icon-4.png'))
        layout.addWidget(self.header)
        layout.addWidget(self.subHeader)
        layout.addWidget(self.standardChoice)
        layout.addWidget(self.personalChoice)
        layout.addStretch()

        self.standardChoice.clicked.connect(self.run_script)
        self.personalChoice.clicked.connect(self.run_script)
        self.showMaximized()

    def run_script(self):
        sender = self.sender()
        if sender == self.standardChoice:
            script_name = 'detection.py'
        elif sender == self.personalChoice:
            script_name = 'detection_learning.py'
        self.standardChoice.hide()
        self.personalChoice.hide()
        self.header.setText('En cours d\'exécution, appuyez sur Q pour quitter')
        self.subHeader.hide()

        self.process = QProcess()
        self.process.finished.connect(self.on_finished)
        self.process.start('python', [script_name])

    def on_finished(self):
        self.standardChoice.show()
        self.personalChoice.show()
        self.subHeader.show()
        self.header.setText("Détecteur de somnolence")
        pid = self.process.processId()

        # find the window with the specified pid
        for win in app.allWindows():
            if win.windowState() == Qt.WindowType.Desktop and win.property("ProcessId") == pid:
                # create a window container and show the video window

                video_window = QWidget.createWindowContainer(win)
                video_window.setWindowFlags(
                    video_window.windowFlags() | Qt.WindowStaysOnTopHint)
                video_window.show()
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(Path("style.css").read_text())
    window = Window()
    window.show()
    sys.exit(app.exec())