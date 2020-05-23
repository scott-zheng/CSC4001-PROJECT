from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import os
import sys
import shutil
import numpy as np
import argparse
import imutils
import time
import dlib
import random
import face_recognition
import queue

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator, QFont
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.uic import loadUi

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# 初始化dlib的面部检测器，并创建面部标识预测器
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r'./shape_predictor_68_face_landmarks.dat')

# 分别获取左眼和右眼的面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # (42, 47)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 41)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]  # (48, 68)

#### util funtions ####
def displayImage(img, qlabel):
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
    qformat = QImage.Format_Indexed8

    if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
        if img.shape[2] == 4:
            # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
            # A: alpha channel
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    
    outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    qlabel.setPixmap(QPixmap.fromImage(outImage))
    qlabel.setScaledContents(True)

def eye_aspect_ratio(eye):
    # 计算两组垂直眼睛地标（x，y）坐标之间的欧氏距离
    A = dist.euclidean(eye[1], eye[5])  # (43, 47)
    B = dist.euclidean(eye[2], eye[4])  # (44, 46)
    # 计算水平眼睛地标（x，y）坐标之间的欧氏距离
    C = dist.euclidean(eye[0], eye[3])  # (42, 45)
    # 计算眼睛的长宽比
    ear = (A + B) / (2.0 * C)
    '''		[1]		[2]
    [0]						[3]
            [5]		[4]
    '''
    # 返回眼睛的长宽比
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[11])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[3], mouth[9])
    D = dist.euclidean(mouth[4], mouth[8])
    E = dist.euclidean(mouth[5], mouth[7])
    F = dist.euclidean(mouth[0], mouth[6])  # 水平欧几里德距离
    '''		[1]		[2]		[3]		[4]		[5]

	[0]												[6]

			[11]	[10]	[9]		[8]		[7]
	'''
    ratio = (A + B + C + D + E) / (5.0 * F)
    return ratio

def left_right_face_ratio(face):
    leftA = dist.euclidean(face[16], face[27])
    leftB = dist.euclidean(face[14], face[30])
    leftC = dist.euclidean(face[12], face[54])
    leftD = dist.euclidean(face[11], face[57])
    rightA = dist.euclidean(face[0], face[27])
    rightB = dist.euclidean(face[2], face[30])
    rightC = dist.euclidean(face[4], face[48])
    rightD = dist.euclidean(face[5], face[57])

    ratioA = rightA / leftA
    ratioB = rightB / leftB
    ratioC = rightC / leftC
    ratioD = rightD / leftD
    ratio = (ratioA + ratioB + ratioC + ratioD) / 4

    return ratio  # 左转大于2.0，右转小于0.5




#### UI Implementation ####

class System(QMainWindow):
    def __init__(self):
        super(System, self).__init__()
        loadUi('./ui/system.ui', self)

        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(668, 538)
        
        pic_1 = QPixmap('./pics/detection.jpg')
        pic_2 = QPixmap('./pics/core.jpg')
        
        self.detectionPartLabel.setPixmap(pic_1)
        self.corePartLabel.setPixmap(pic_2)
        self.detectionPartLabel.setScaledContents(True)
        self.corePartLabel.setScaledContents(True)

        self.haveTryButton.clicked.connect(self.detectionPart)

        self.coreEnterButton.clicked.connect(self.corePart)

    def detectionPart(self):
        self.detectionPartWidget = DetectionPartWidget()
        self.detectionPartWidget.setWindowModality(Qt.ApplicationModal)
        self.detectionPartWidget.show()

    def corePart(self):
        self.corePartWidget = CorePartWidget()
        self.corePartWidget.setWindowModality(Qt.ApplicationModal)
        self.corePartWidget.show()

    def closeEvent(self, event):
        result = QMessageBox.question(self, "Quit", "<font size=4>Do you want to exit?</font>", QMessageBox.Yes | QMessageBox.No)
        if(result == QMessageBox.Yes):
            # clear all the photos captured in /tmp
            shutil.rmtree('./User_info/tmp')
            os.mkdir('./User_info/tmp')

            # exit all
            sys.exit(0)
            event.accept()
        else:
            event.ignore()

class DetectionPartWidget(QWidget):
    def __init__(self):
        super(DetectionPartWidget, self).__init__()
        loadUi('./ui/detectionPartWidget.ui', self)

        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(686, 385)

        # initialize camera
        self.cap = cv2.VideoCapture()

        # Realtime detect
        self.RealtimeDetectButton.clicked.connect(self.realTimeDetect)

        # Photo detect
        self.PhotoDetectButton.clicked.connect(self.choosePhoto)

        # Clear detection frame
        self.clearButton.clicked.connect(self.clearFrame)

        self.timer = QTimer(self)  # initilize a timer
        self.faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def choosePhoto(self):
        # if camera is open, then swith it off.
        if (self.cap.isOpened()):
            if self.timer.isActive():
                self.timer.stop()
            self.cap.release()
            self.displayLabel.clear()        

        imgName, imgType = QFileDialog.getOpenFileName(self,
                                    "Choose a photo",
                                    "./",
                                    "Image Files(*.jpg *.jpeg *.png)")

        if (imgName == ''):
            return

        # opencv operations
        img = self.photoDetect(imgName)

        # output the modified image and display on the screen
        displayImage(img, self.displayLabel)

        # img = QPixmap(imgName).scaled(self.displayLabel.width(), self.displayLabel.height())
        # self.displayLabel.setPixmap(img)

    def photoDetect(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceDetect.detectMultiScale(gray, 1.1, 10)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img
    
    def realTimeDetect(self):
        self.cap.open(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox('Fail to use camera.')
            self.cap.release()
        else:
            self.timer.start(1)
            self.timer.timeout.connect(self.updateFrame)
    
    def clearFrame(self):
        self.displayLabel.clear()

        if (self.cap.isOpened()):
            if self.timer.isActive():
                self.timer.stop()
            self.cap.release()
    
    def updateFrame(self):
        if self.cap.isOpened():  
            ret, img = self.cap.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceDetect.detectMultiScale(gray, 1.1, 10)

            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            displayImage(img, self.displayLabel)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

class CorePartWidget(QWidget):
    def __init__(self):
        super(CorePartWidget, self).__init__()
        loadUi('./ui/corePartWidget.ui', self)

        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(462, 421)

        # initialize camera
        self.cap = cv2.VideoCapture()

        # Data register
        self.registerButton.clicked.connect(self.dataRegister)

        # Face recognize
        self.recognizeButton.clicked.connect(self.recognize)

        ############
        self.timer = QTimer(self)  # initilize a timer
        self.faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   
    def recognize(self):
        self.resetRecogInfo()

        # Live test
        self.liveDetectionWidget = liveDetectionWidget(self)
        self.liveDetectionWidget.setWindowModality(Qt.ApplicationModal)
        self.liveDetectionWidget.show()

        # face recognize part and output info
        self.liveDetectionWidget.signal.connect(self.liveDetectionSignalEvent)

    def dataRegister(self):
        self.resetRecogInfo()

        self.userInfoDialog = UserInfoDialog()
        self.userInfoDialog.exec()

        # receive signal
        self.userInfoDialog.signal.connect(self.userInfoDialogSignalEvent)
    
    def liveDetectionSignalEvent(self, isLive):
        if isLive:
            # recognize
            self.realTimeRecogWidget = realTimeRecogWidget()
            self.realTimeRecogWidget.setWindowModality(Qt.ApplicationModal)
            self.realTimeRecogWidget.show()
        
            # output info
            self.realTimeRecogWidget.signal.connect(self.realTimeRecogSignalEvent)      
        else:
            pass
    
    def realTimeRecogSignalEvent(self, userInfo):
        print(userInfo)
        if userInfo:
            self.NameLineEdit.setText(userInfo[0])
            if userInfo[1] == 'male':
                self.maleButton.setChecked(True)
            else:
                self.femaleButton.setChecked(True)
            self.AgeLineEdit.setText(userInfo[2])
            self.IDLineEdit.setText(userInfo[3])

    def userInfoDialogSignalEvent(self, signal):
        if (signal==True): ## all info has been collected
            # write user info data into files
            name, age, id_number =\
            self.userInfoDialog.NameLineEdit.text(), self.userInfoDialog.AgeLineEdit.text(), self.userInfoDialog.IDLineEdit.text()
            gender = 'male' if self.userInfoDialog.maleButton.isChecked() else 'female'

            print('name: %s, age: %s, id: %s, gender: %s' % (name, age, id_number, gender))

            ## store the info into User_info
            os.mkdir('./User_info/%s/' % id_number)
            with open('./User_info/%s/identity_info.txt' % id_number, 'w') as f:
                f.write(name+'\n')
                f.write(gender+'\n')
                f.write(age+'\n')

            with open('./User_info/id_table.txt', 'a+') as f:
                f.write(id_number+'\n')
            
            # move the face photos in ./tmp into permanent ./stored
            shutil.copytree('./User_info/tmp', './User_info/%s/user_faces' % id_number)
            shutil.rmtree('./User_info/tmp')
            os.mkdir('./User_info/tmp')

            # announce the success
            success = QMessageBox.information(self, "Register", "<font size=4>You have successfully registered!</font>")

    def resetRecogInfo(self):
        # swith off and clear the line edits
        for line_edit in [self.NameLineEdit, self.AgeLineEdit, self.IDLineEdit]:
            line_edit.setReadOnly(True)
            line_edit.clear()
        self.maleButton.setChecked(True)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()

        # clear all the photos captured in /tmp
        shutil.rmtree('./User_info/tmp')
        os.mkdir('./User_info/tmp')

        event.accept()

class realTimeRecogWidget(QWidget):
    signal = pyqtSignal(list)
    def __init__(self):
        super(realTimeRecogWidget, self).__init__()
        loadUi('./ui/realTImeRecogWidget.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(661, 411)

        self.hasRecognized = False
        self.paused = False

        self.cap = cv2.VideoCapture(0)

        self.captureButton.clicked.connect(self.capture)

        self.timer = QTimer(self)
        self.timer.start(1)
        self.timer.timeout.connect(self.updateFrame)
    
    def capture(self):
        if not self.paused:
            self.captureButton.setText('RETRY')
            self.realTimeRecog()
        else:
            self.captureButton.setText('CAPTURE')
            self.timer.start(1)

        self.paused = not self.paused

    def updateFrame(self):
        if self.cap.isOpened():
            ret, img = self.cap.read()
            img = cv2.flip(img, 1)
            displayImage(img, self.cameraLabel)

    def realTimeRecog(self):
        ret, frame = self.cap.read()
        unknown_img = frame
        locations = face_recognition.face_locations(unknown_img)
        encodings = face_recognition.face_encodings(unknown_img)
        # 对每一帧中的所有人脸进行比较
        for (top, right, bottom, left), unknown_face_encoding in zip(locations, encodings):
            find_tag = 0
            cv2.rectangle(unknown_img, (left, top), (right, bottom), (0, 0, 255), 2)
            
            user_list = []
            # read out user list
            with open('./User_info/id_table.txt', 'r') as f:
                for line in f:
                    user_list.append(line.replace('\n', ''))
            
            for id_number in user_list:
                user_faces_path = './User_info/' + id_number + '/user_faces'
                user_faces_path_list = os.listdir(user_faces_path)

                for face_name in user_faces_path_list:  # 遍历一个人的所有脸
                    single_face_path = user_faces_path + '/' + face_name
                    known_img = cv2.imread(single_face_path)
                    known_img_encoding = face_recognition.face_encodings(known_img)
                    match_result = face_recognition.compare_faces(known_img_encoding, unknown_face_encoding)

                    if match_result == [True]:
                        with open('./User_info/' + id_number + '/identity_info.txt', 'r') as f:
                            user_info_list = f.readlines()
                            name = user_info_list[0].replace('\n', '')
                            gender = user_info_list[1].replace('\n', '')
                            age = user_info_list[2].replace('\n', '')

                        unknown_img = cv2.flip(unknown_img, 1)
                        cv2.putText(unknown_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 0, 255), 3)
                        find_tag = 1
                        break

                if find_tag == 1:
                    self.signal.emit([name, gender, age, id_number])
                    self.hasRecognized = True
                    displayImage(unknown_img, self.cameraLabel)
        #             time.sleep(2)
                    self.timer.stop()
                    break
    
    def closeEvent(self, event):
        if not self.hasRecognized:
            self.signal.emit([])

        if self.timer.isActive:
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()
        

class UserInfoDialog(QDialog):
    signal = pyqtSignal(bool)

    def __init__(self):
        super(UserInfoDialog, self).__init__()
        loadUi('./ui/userInfoDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(425, 300)

        self.NameLineEdit.setFocus()
        
        name_regx = QRegExp('^[ A-Za-z]{1,24}$')
        name_validator = QRegExpValidator(name_regx, self.NameLineEdit)
        self.NameLineEdit.setValidator(name_validator)

        id_regx = QRegExp('^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9X]$')
        id_validator = QRegExpValidator(id_regx, self.IDLineEdit)
        self.IDLineEdit.setValidator(id_validator)

        age_regx = QRegExp('^((1[0-1])|[1-9])?\d$')
        age_validator = QRegExpValidator(age_regx, self.AgeLineEdit)
        self.AgeLineEdit.setValidator(age_validator)

        self.nextStepButton.clicked.connect(self.checkToNextStep)
    
    # check user input and do the next step
    def checkToNextStep(self):
        if not (self.NameLineEdit.hasAcceptableInput() and
                self.IDLineEdit.hasAcceptableInput() and
                self.AgeLineEdit.hasAcceptableInput()):
            self.msgLabel.setText('<font color=red>Your input is invalid. Fail to submit. Please try again! </font>')
        elif (os.path.exists("./User_info/%s" % self.IDLineEdit.text())):
            self.msgLabel.setText('<font color=red>You have already registered your information! </font>')
        else:
            # 活体检测
            self.liveDetectionWidget = liveDetectionWidget(self)
            self.liveDetectionWidget.setWindowModality(Qt.ApplicationModal)
            self.liveDetectionWidget.show()
            self.hide()
            # receive signal
            self.liveDetectionWidget.signal.connect(self.liveDetectionSignalEvent)
            
    def liveDetectionSignalEvent(self, isLive):
        if not isLive:
            self.show()
            
        else:
            self.msgLabel.setText('')
            self.faceCollectWidget = FaceCollectWidget(self)
            self.faceCollectWidget.setWindowModality(Qt.ApplicationModal)
            self.faceCollectWidget.show()
            self.hide()

            # receive signal
            self.faceCollectWidget.signal.connect(self.faceCollectWidgetSignalEvent)

    def faceCollectWidgetSignalEvent(self, signal):
        if (signal==True): ## face collection has finished
            # send signal
            self.signal.emit(True)
            self.close()
        else:
            self.signal.emit(False)
            self.show()

class liveDetectionWidget(QWidget):
    signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(liveDetectionWidget, self).__init__()
        loadUi('./ui/liveDetectionWidget.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setFixedSize(660, 411)
        self.timer = QTimer(self)
        self.capture = cv2.VideoCapture()
        self.finish_flag = 0
        
        self.startButton.clicked.connect(self.liveDetection)
        self.random_number = random.randint(1, 2)
    
    def liveDetection(self):
        self.startButton.setEnabled(False)

        self.COUNTER = 0
        self.TOTAL = 0
        self.OPEN_MOUTH_COUNTER = 0
        self.MOUTH_TOTAL = 0
        self.TURN_LEFT_TOTAL = 0
        self.TURN_LEFT_COUNTER = 0
        self.TURN_RIGHT_TOTAL = 0
        self.TURN_RIGHT_COUNTER = 0


        # 启动视频流线程
        print("[INFO] starting video stream thread...")
        fileStream = True
        self.capture.open(0)
        fileStream = False
        
        self.timer.start(1)
        self.timer.timeout.connect(self.update)

    def update(self):
        textColor = (255, 0, 0)
        ret, frame = self.capture.read()
        frame = imutils.resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.finish_flag == 0:
            # 在灰度框种检测人脸
            rects = detector(gray, 0)

            # 循环人脸检测
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 提取左眼和右眼坐标，然后使用坐标计算两只眼睛的眼睛纵横比
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mouthRatio = mouth_aspect_ratio(mouth)
                leftRightRatio = left_right_face_ratio(shape)

                # 平均左右眼睛纵横比
                ear = (leftEAR + rightEAR) / 2.0

                # 计算左眼，右眼和嘴的凸包，然后可视化每只眼睛
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                print("leftRightRatio:", leftRightRatio)
                if mouthRatio > 0.7:
                    self.OPEN_MOUTH_COUNTER += 1
                else:
                    if self.OPEN_MOUTH_COUNTER >= 2:
                        self.MOUTH_TOTAL += 1
                    self.OPEN_MOUTH_COUNTER = 0
                if leftRightRatio >= 2.0:
                    self.TURN_LEFT_COUNTER += 1
                elif leftRightRatio <= 0.5:
                    self.TURN_RIGHT_COUNTER += 1
                else:
                    if self.TURN_LEFT_COUNTER >= 2:
                        self.TURN_LEFT_TOTAL += 1
                    if self.TURN_RIGHT_COUNTER >= 2:
                        self.TURN_RIGHT_TOTAL += 1
                    self.TURN_LEFT_COUNTER = 0
                    self.TURN_RIGHT_COUNTER = 0

                # 检查眼睛宽高比是否低于眨眼阈值，如果是，则增加眨眼帧计数器
                if ear < EYE_AR_THRESH:
                    self.COUNTER += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # 闭眼时间大于等于2帧图片
                    if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                        # 检测完成后，眨眼重新检测
                        if self.MOUTH_TOTAL >= 1 and self.TURN_LEFT_TOTAL >= 1 and self.TURN_RIGHT_TOTAL >= 1:
                            self.TURN_LEFT_TOTAL = 0
                            self.TURN_RIGHT_TOTAL = 0
                            self.MOUTH_TOTAL = 0

                            self.random_number = random.randint(1, 6)

                    # reset the eye frame counter
                    self.COUNTER = 0

                if self.random_number == 1:
                    if self.TURN_LEFT_TOTAL > 0:
                        if self.TURN_RIGHT_TOTAL > 0:
                            if self.MOUTH_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Open Mouth", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Turn Right Face", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.MOUTH_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Turn Left Face", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.MOUTH_TOTAL = 0
                        self.TURN_RIGHT_TOTAL = 0
                elif self.random_number == 2:
                    if self.MOUTH_TOTAL > 0:
                        if self.TURN_RIGHT_TOTAL > 0:
                            if self.TURN_LEFT_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Turn Left Face", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Turn Right Face", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.TURN_LEFT_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Open Mouth", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.TURN_LEFT_TOTAL = 0
                        self.TURN_RIGHT_TOTAL = 0
                elif self.random_number == 3:
                    if self.TURN_RIGHT_TOTAL > 0:
                        if self.TURN_LEFT_TOTAL > 0:
                            if self.MOUTH_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Open Mouth", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Turn Left Face", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.MOUTH_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Turn Right Face", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.MOUTH_TOTAL = 0
                        self.TURN_LEFT_TOTAL = 0
                elif self.random_number == 4:
                    if self.MOUTH_TOTAL > 0:
                        if self.TURN_LEFT_TOTAL > 0:
                            if self.TURN_RIGHT_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Turn Right Face", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Turn Left Face", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.TURN_RIGHT_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Open Mouth", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.TURN_RIGHT_TOTAL = 0
                        self.TURN_LEFT_TOTAL = 0
                elif self.random_number == 5:
                    if self.TURN_LEFT_TOTAL > 0:
                        if self.MOUTH_TOTAL > 0:
                            if self.TURN_RIGHT_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Turn Right Face", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Open Mouth", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.TURN_RIGHT_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Turn Left Face", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.TURN_RIGHT_TOTAL = 0
                        self.MOUTH_TOTAL = 0
                elif self.random_number == 6:
                    if self.TURN_RIGHT_TOTAL > 0:
                        if self.MOUTH_TOTAL > 0:
                            if self.TURN_LEFT_TOTAL > 0:
                                self.finish_flag = 1
                                QMessageBox().information(self, 'Message', '<font size=4>You have passed live test!</font>')
                            else:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Turn Left Face", (200, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        else:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, "Open Mouth", (200, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                            self.TURN_LEFT_TOTAL = 0
                    else:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, "Turn Right Face", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, textColor, 4)
                        self.TURN_LEFT_TOTAL = 0
                        self.MOUTH_TOTAL = 0

            if len(rects) == 0:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "No Face Detected!", (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)

            # show the frame
            displayImage(frame, self.cameraLabel)
        else:
            self.signal.emit(True)
            self.close()
    
    def closeEvent(self, event):
        if self.timer.isActive:
            self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        event.accept()
    
class FaceCollectWidget(QWidget):
    signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(FaceCollectWidget, self).__init__()
        loadUi('./ui/faceCollectWidget.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setFixedSize(662, 601)
        # self.font = QFont('Calibri', 30)

        self.cap = cv2.VideoCapture()
        self.timer = QTimer(self)
        self.faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.isFaceDetected = False # for deciding wether face detection is open
        self.detectedFaceNum = 0 # flag for ensuring only one face detected - 0:no face; 1:one face; 2:more faces
        self.faceFrameCount = 0
        self.hasFinished = False

        self.switchButton.clicked.connect(self.switchCamera)
        self.enableFaceDetectButton.clicked.connect(self.switchFaceDetect)

        self.captureButton.clicked.connect(self.captureFaceFrame)
        self.finishButton.clicked.connect(self.finish)
      
    def switchCamera(self):
        if not self.cap.isOpened():

            # display camera capture on to the central label
            self.cap.open(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox('Fail to use camera.')
                self.cap.release()
            else:
                self.timer.start(1)
                self.timer.timeout.connect(self.updateFrame)

            # enable the face detection button
            self.enableFaceDetectButton.setEnabled(True)

            # change the switch button to be 'turn off'
            self.switchButton.setText('Turn off')

        else:
            # stop timer
            if self.timer.isActive():
                self.timer.stop()

            # stop camera
            self.cap.release()
            self.faceDetectCaptureLabel.clear()

            # disable the face detection button
            self.enableFaceDetectButton.setEnabled(False)
            
            # change the switch button to be 'turn on'
            self.switchButton.setText('Turn on')

    def switchFaceDetect(self):
        if not self.isFaceDetected: # if face detection is off
            self.isFaceDetected = True
            self.enableFaceDetectButton.setText('Stop Face Detection')
        else:
            self.isFaceDetected = False
            self.enableFaceDetectButton.setText('Open Face Detection')
    
    def captureFaceFrame(self):
        if self.faceFrameCount < 10:
            if self.isFaceDetected:
                if self.detectedFaceNum == 1:
                    ## capture face frame
                    ret, frame = self.cap.read()
                    img = frame
                    img = cv2.flip(img, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.faceDetect.detectMultiScale(gray, 1.1, 10)

                    # store face capture
                    for x, y, w, h in faces:
                        cv2.imwrite('./User_info/tmp/'+str(self.faceFrameCount+1)+'.jpg', img[y:y + h, x:x + w])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # display count on LCD
                    self.faceFrameCount += 1
                    self.faceFrameCountLcdNum.display(self.faceFrameCount)
                
                # no face
                elif self.detectedFaceNum == 0:
                    msg = QMessageBox().warning(self, 'Message', '<font size=4 color=red>No face detected!</font>')
                
                # more than one faces 
                else:
                    msg = QMessageBox().warning(self, 'Message', '<font size=4 color=red>More than one faces detected!</font>')
            else:
                msg = QMessageBox().warning(self, 'Message', '<font size=4 color=red>Open face detection first!</font>')
            # msg.setFont(self.font)
        else:
            msg = QMessageBox().information(self, 'Message', '<font size=4>You have finished 10 captures.</font>')

    def finish(self):
        if self.faceFrameCount < 10:
            msg = QMessageBox.warning(self, 'Message', '<font size=4 color=red>You have not taken 10 captures!</font>')
        else:
            # send signal
            self.signal.emit(True)
            self.hasFinished = True
            self.close()

    def updateFrame(self):
        if self.cap.isOpened():
            ret, img = self.cap.read()
            img = cv2.flip(img, 1)

            if self.isFaceDetected:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.faceDetect.detectMultiScale(gray, 1.1, 10)
                
                for x, y, w, h in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                self.detectedFaceNum = len(faces)
                # give warning if more than one face appear
                if len(faces) >= 2:
                    cv2.putText(img, 'More than one faces detected!', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                elif len(faces) == 0:
                    cv2.putText(img, 'No face detected!', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            displayImage(img, self.faceDetectCaptureLabel)

    def closeEvent(self, event):
        if self.hasFinished:
            if self.timer.isActive():
                self.timer.stop()
            if self.cap.isOpened():
                self.cap.release()
        else:
            result = QMessageBox.question(self, "Quit", "<font size=4>Sure to cancel face collection?</font>", QMessageBox.Yes | QMessageBox.No)
            if(result == QMessageBox.Yes):
                # send signal
                self.signal.emit(False)
                # clear all the photos captured in /tmp
                shutil.rmtree('./User_info/tmp')
                os.mkdir('./User_info/tmp')

                # reset other settings
                if self.timer.isActive():
                    self.timer.stop()
                if self.cap.isOpened():
                    self.cap.release()

                event.accept()
            else:
                event.ignore()
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = System()
    window.show()
    sys.exit(app.exec())