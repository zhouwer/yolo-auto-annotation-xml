# -*- coding: utf-8 -*-

import sys  # 导入系统
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton,QLabel,QTextEdit,QFileDialog,QHBoxLayout,QVBoxLayout,QSplitter,QComboBox,QSpinBox
from PyQt5.Qt import QWidget, QColor,QPixmap,QIcon,QSize,QCheckBox
from PyQt5 import QtCore, QtGui
from Paintboard import PaintBoard
import colorsys
import os
from timeit import default_timer as timer
import cv2
import os
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import xml.dom.minidom
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import sys
import argparse
from yoloxml import YOLO, detect_video1
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import pickle
import  keras
import os
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
#from gcforest.gcforest import GCForest
import keras.backend.tensorflow_backend as KTF

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
config = tf.ConfigProto()

config.gpu_options.allow_growth=True #不全部占满显存, 按需分配

session = tf.Session(config=config) # 设置session KTF.set_session(sess)

class FirstUi(QMainWindow):  # 第一个窗口类
    def __init__(self):
        super(FirstUi, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(self.width(), self.height())
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 500)  # 设置窗口大小
        self.setWindowTitle('自动标注')  # 设置窗口标题
        self.btn = QPushButton('视频点这个', self)  # 设置按钮和按钮名称
        self.btn.setGeometry(245, 100, 150, 50)  # 前面是按钮左上角坐标，后面是窗口大小
        self.btn.clicked.connect(self.slot_btn_function)  # 将信号连接到槽
        self.btn2 = QPushButton('图片组点这个', self)
        self.btn2.setGeometry(245, 200,150,50)
        self.btn2.clicked.connect(self.slot_btn2_function)
        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(245, 300, 150, 50)
        self.btn_exit.clicked.connect(self.Quit)
        self.label_name = QLabel('hehe', self)
        self.label_name.setGeometry(460, 410, 200, 30)
        self.label_name = QLabel('啊哈', self)
        self.label_name.setGeometry(460, 440, 200, 30)

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()  # 隐藏此窗口
        #self.setHidden(True)
        self.s = danzhen() # 将第二个窗口换个名字
        #self.s = write_num()
        self.s.show()  # 经第二个窗口显示出来

    def slot_btn2_function(self):
        self.hide()  # 隐藏此窗口
        self.s = picture_num()
        self.s.show()

#省略
class write_num(QWidget):
    def __init__(self):
        super(write_num, self).__init__()
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 600)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("识别啊啊")

        self.label_name = QLabel('哔哩哔哩大学', self)
        self.label_name.setGeometry(500, 5, 120, 30)

        self.label_name = QLabel('知识学院', self)
        self.label_name.setGeometry(500, 35, 100, 30)

        self.label_name = QLabel('野生技术协会', self)
        self.label_name.setGeometry(500, 65, 100, 30)

        self.label_name = QLabel('南岛鹋', self)
        self.label_name.setGeometry(500, 95, 100, 30)

        self.edit = QTextEdit(self)
        self.edit.setGeometry(510, 160, 110, 60)

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__btn_Recognize = QPushButton("开始识别")
        self.__btn_Recognize.setParent(self)
        self.__btn_Recognize.clicked.connect(self.on_btn_Recognize_Clicked)
        sub_layout.addWidget(self.__btn_Recognize)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面
        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_return = QPushButton("返回")
        self.__btn_return.setParent(self)  # 设置父对象为本界面
        self.__btn_return.clicked.connect(self.slot_btn_function)
        sub_layout.addWidget(self.__btn_return)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(20)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange)  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.jpg')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        print(savePath[0])

    def Quit(self):
        self.close()

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def on_btn_Recognize_Clicked(self):
        print(218)
        #config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        print(220)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.8
        #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        savePath = "text1.png"   #必须png
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)
        image = Image.open(savePath)
        # image = image.convert('L')
        # 109 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=720x576 at 0x1DA65C3E160>
        print(344, image)  # ok!

        # image_data = np.array(image, dtype='float32')
        # img = img.convert('L')
        # print(image_data.shape)
        model_image_size = (28, 28)
        print(354, image.size)

        def letterbox_image(image, size):
            '''resize image with unchanged aspect ratio using padding'''
            print(357)
            iw, ih = image.size
            print(359)
            w, h = size
            print(iw, ih, w, h)
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            print(366, nw, nh)
            image = image.resize((nw, nw), Image.BICUBIC)
            print(368)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            return new_image

        print(372)
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        boxed_image = boxed_image.convert('L')
        print(374, boxed_image, boxed_image.size)
        image = np.array(boxed_image)  # ????????????????????
        print(376, image.shape)
        # image=np.array(image)
        # img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        # img = img.convert('L')
        # x = keras.preprocessing.image.img_to_array(img)
        # x = abs(255-x)
        # x = x.reshape(28,28)
        # x = np.expand_dims(x, axis=0)
        x = image / 255.0
        print(384, x.shape)
        # new_model = keras.models.load_model('my_model.h5')
        print(386)
        # x= np.random.rand(28,28)
        x = x.reshape(1, -1)
        print(x.shape)
        # gc = GCForest(config)

        with open("C:/Users/齐天大圣/Desktop/税/gc -forest,原始/gcForest-master/examples/model2020724.pkl", "rb") as f:
            gc = pickle.load(f)
        print(375)

        prediction = gc.predict(x)

        # prediction = new_model.predict(x)
        print(349)
        print(prediction)
        #image.save(savePath)
        #image = Image.open(savePath)
        #model_image_size = (28, 28)


        #boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        #image = np.array(boxed_image, dtype='float32')
        #img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        #img = img.convert('L')
        #x = keras.preprocessing.image.img_to_array(img)
        #x = abs(255-x)
        #x = x.reshape(28,28)
        #x=image/255
        #x = np.expand_dims(x, axis=0)
        #x=x/255.0
        #print(252)
        #new_model = keras.models.load_model('my_model.h5')
        #print(254)
        #x = np.random.rand(28, 28)
        #print(256,x.shape)
        #prediction = new_model.predict(x)
        #print(258)
        #output = np.argmax(prediction, axis=1)
        #print("手写数字识别为：" + str(output[0]))
        self.edit.setText('识别的手写数字为:' + str(prediction))

    def slot_btn_function(self):
        self.hide()  # 隐藏此窗口
        self.f = FirstUi()  # 将第一个窗口换个名字
        self.f.show()  # 将第一个窗口显示出来


class danzhen(QWidget):
    def __init__(self):
        super(danzhen, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 520)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('视频生成xml')
        # self.label_name1 = QLabel('哔哩哔哩大学', self)
        # self.label_name1.setGeometry(500, 20, 120, 35)

        # self.label_name2 = QLabel('知识学院', self)
        # self.label_name2.setGeometry(500, 60, 100, 35)

        # self.label_name3 = QLabel('野生技术协会', self)
        # self.label_name3.setGeometry(500, 100, 100, 35)

        # self.label_name4 = QLabel('南岛鹋', self)
        # self.label_name4.setGeometry(500, 140, 100, 35)

        self.label_name5 = QLabel('🐱🐟🐕🦈🐅🦁🐘', self)
        self.label_name5.setGeometry(10, 20, 480, 480)
        self.label_name5.setStyleSheet("QLabel{background:pink;}"
                                       "QLabel{color:rgb(100,100,100,960);font-size:50px;font-weight:bold;font-family:楷体;}"
                                       )
        self.label_name5.setAlignment(QtCore.Qt.AlignCenter)

        self.edit = QTextEdit(self)
        self.edit.setGeometry(500, 220, 100, 60)

        self.btn_select = QPushButton('选视频', self)
        self.btn_select.setGeometry(500, 320, 100, 30)
        self.btn_select.clicked.connect(self.select_video)

        self.btn_dis = QPushButton('生成图片、xml', self)
        self.btn_dis.setGeometry(500, 370, 100, 30)
        self.btn_dis.clicked.connect(self.on_btn_Recognize_Clicked)

        self.btn = QPushButton('返回', self)
        self.btn.setGeometry(500, 420, 100, 30)
        self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(500, 470, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_video(self):
        print(401)
        global absolute_path
        print(403)
        #print(405,fname)
        absolute_path = QFileDialog.getOpenFileName(self, "Open file", "", "*.mp4;;*.png;;All Files(*)")
        #print(386, self.label_name5.width(), self.label_name5.height(), imgName, imgType)
        print(408, absolute_path[0])





    def on_btn_Recognize_Clicked(self):
        global absolute_path
        print(417,absolute_path[0])
        vc = cv2.VideoCapture(absolute_path[0])
        c = 0
        rval = vc.isOpened()
        while rval:  # 循环读取视频帧
            c = c + 1
            rval, frame = vc.read()
            # print(rval, frame)
            pic_path = r'meizhentu/'
            isExists = os.path.exists(pic_path)
            if not isExists:
                # 如果不存在则创建目录
                os.makedirs(pic_path)
                print("创好了")
            else:
                # 如果目录存在则不创建，并提示目录已存在
                #print("已存")
                fftefgg45 = 1
            if rval:
                cv2.imwrite(pic_path + str(c) + '.png', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                cv2.waitKey(1)
            else:
                fftefgg=1
        vc.release()


        print('447--生成单帧图完成')
        zaisheng_path = r'kongxml/'
        #isExists = os.path.exists(pic_path+zaisheng_path)
        isExists = os.path.exists(zaisheng_path)
        if not isExists:
            # 如果不存在则创建目录
            #os.makedirs(pic_path+zaisheng_path)
            os.makedirs(zaisheng_path)
            print("创好了")
        else:
            # 如果目录存在则不创建，并提示目录已存在
            #print("已存")
            ekdiu=0
        print('454--创建完了每张图的空xml')


        print(457)

        fname1 = isExists
        print(fname1)

        ## "model_path": 'logs/003/trained_weights_final.h5',
        ## "model_path": 'voc_weights.h5',
        class YOLO(object):
            _defaults = {

                "model_path": 'model_data/tiny_yolo_weights.h5',

                "anchors_path": 'model_data/yolo_anchors.txt',
                "classes_path": 'model_data/coco_classes.txt',
                "score": 0.2,  # 0.3
                "iou": 0.25,  # 0.45
                "model_image_size": (416, 416),
                "gpu_num": 1,
            }

            @classmethod
            def get_defaults(cls, n):
                if n in cls._defaults:
                    return cls._defaults[n]
                else:
                    return "Unrecognized attribute name '" + n + "'"

            def __init__(self, **kwargs):
                self.__dict__.update(self._defaults)  # set up default values
                self.__dict__.update(kwargs)  # and update with user overrides
                self.class_names = self._get_class()
                self.anchors = self._get_anchors()
                self.sess = K.get_session()
                self.boxes, self.scores, self.classes = self.generate()

            def _get_class(self):
                classes_path = os.path.expanduser(self.classes_path)
                with open(classes_path) as f:
                    class_names = f.readlines()
                class_names = [c.strip() for c in class_names]
                return class_names

            def _get_anchors(self):
                anchors_path = os.path.expanduser(self.anchors_path)
                with open(anchors_path) as f:
                    anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                return np.array(anchors).reshape(-1, 2)

            def generate(self):
                model_path = os.path.expanduser(self.model_path)
                assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

                # Load model, or construct model and load weights.
                num_anchors = len(self.anchors)
                num_classes = len(self.class_names)
                is_tiny_version = num_anchors == 6  # default setting
                try:
                    self.yolo_model = load_model(model_path, compile=False)
                except:
                    self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                        if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
                    self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
                else:
                    assert self.yolo_model.layers[-1].output_shape[-1] == \
                           num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                        'Mismatch between model and given anchor and class sizes'

                print('{} model, anchors, and classes loaded.'.format(model_path))

                # Generate colors for drawing bounding boxes.
                hsv_tuples = [(x / len(self.class_names), 1., 1.)
                              for x in range(len(self.class_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)  # Fixed seed for consistent colors across runs.
                np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
                np.random.seed(None)  # Reset seed to default.

                # Generate output tensor targets for filtered bounding boxes.
                self.input_image_shape = K.placeholder(shape=(2,))
                if self.gpu_num >= 2:
                    self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
                boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                   len(self.class_names), self.input_image_shape,
                                                   score_threshold=self.score, iou_threshold=self.iou)
                return boxes, scores, classes


            def detect_xml(self, image, lxml):
                start = timer()
                print(104)
                if self.model_image_size != (None, None):

                    assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'

                    assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'

                    boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))

                else:

                    new_image_size = (image.width - (image.width % 32),

                                      image.height - (image.height % 32))

                    boxed_image = letterbox_image(image, new_image_size)

                image_data = np.array(boxed_image, dtype='float32')

                print(image_data.shape)

                image_data /= 255.

                image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

                out_boxes, out_scores, out_classes = self.sess.run(

                    [self.boxes, self.scores, self.classes],

                    feed_dict={

                        self.yolo_model.input: image_data,

                        self.input_image_shape: [image.size[1], image.size[0]],

                        K.learning_phase(): 0

                    })
                print('142', out_boxes, out_scores, out_classes)  # print("s447",label, (left, top), (right, bottom))
                doc = xml.dom.minidom.Document()

                annotation = doc.createElement('annotation')

                doc.appendChild(annotation)

                folder = doc.createElement('folder')

                folder_text = doc.createTextNode('blank')

                folder.appendChild(folder_text)

                annotation.appendChild(folder)

                filename = doc.createElement('filename')

                filename_text = doc.createTextNode('blank')

                filename.appendChild(filename_text)

                annotation.appendChild(filename)

                path = doc.createElement('path')

                path_text = doc.createTextNode('blank')

                path.appendChild(path_text)

                annotation.appendChild(path)

                source = doc.createElement('source')

                databass = doc.createElement('databass')

                databass_text = doc.createTextNode('Unknown')

                source.appendChild(databass)

                databass.appendChild(databass_text)

                annotation.appendChild(source)

                size = doc.createElement('size')

                width = doc.createElement('width')

                width_text = doc.createTextNode(str(image.width))

                height = doc.createElement('height')

                height_text = doc.createTextNode(str(image.height))

                depth = doc.createElement('depth')

                depth_text = doc.createTextNode('3')

                size.appendChild(width)

                width.appendChild(width_text)

                size.appendChild(height)

                height.appendChild(height_text)

                size.appendChild(depth)

                depth.appendChild(depth_text)

                annotation.appendChild(size)

                segmented = doc.createElement('segmented')

                segmented_text = doc.createTextNode('0')

                segmented.appendChild(segmented_text)

                annotation.appendChild(segmented)

                for i, c in reversed(list(enumerate(out_classes))):

                    predicted_class = self.class_names[c]

                    if str(predicted_class) == 'screen':
                        box = out_boxes[i]

                        lobject = doc.createElement('object')

                        name = doc.createElement('name')

                        name_text = doc.createTextNode(str(predicted_class))

                        pose = doc.createElement('pose')

                        pose_text = doc.createTextNode('Unspecified')

                        truncated = doc.createElement('truncated')

                        truncated_text = doc.createTextNode('0')

                        difficult = doc.createElement('difficult')

                        difficult_text = doc.createTextNode('0')

                        name.appendChild(name_text)

                        lobject.appendChild(name)

                        pose.appendChild(pose_text)

                        lobject.appendChild(pose)

                        truncated.appendChild(truncated_text)

                        lobject.appendChild(truncated)

                        difficult.appendChild(difficult_text)

                        lobject.appendChild(difficult)

                        top, left, bottom, right = box

                        top = max(0, np.floor(top + 0.5).astype('int32'))

                        left = max(0, np.floor(left + 0.5).astype('int32'))

                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        bndbox = doc.createElement('bndbox')

                        xmin = doc.createElement('xmin')

                        xmin_text = doc.createTextNode(str(left))

                        ymin = doc.createElement('ymin')

                        ymin_text = doc.createTextNode(str(top))

                        xmax = doc.createElement('xmax')

                        xmax_text = doc.createTextNode(str(right))

                        ymax = doc.createElement('ymax')

                        ymax_text = doc.createTextNode(str(bottom))

                        xmin.appendChild(xmin_text)

                        bndbox.appendChild(xmin)

                        ymin.appendChild(ymin_text)

                        bndbox.appendChild(ymin)

                        xmax.appendChild(xmax_text)

                        bndbox.appendChild(xmax)

                        ymax.appendChild(ymax_text)

                        bndbox.appendChild(ymax)

                        lobject.appendChild(bndbox)

                        annotation.appendChild(lobject)
                    if str(predicted_class) == 'icon':
                        box = out_boxes[i]

                        lobject = doc.createElement('object')

                        name = doc.createElement('name')

                        name_text = doc.createTextNode(str(predicted_class))

                        pose = doc.createElement('pose')

                        pose_text = doc.createTextNode('Unspecified')

                        truncated = doc.createElement('truncated')

                        truncated_text = doc.createTextNode('0')

                        difficult = doc.createElement('difficult')

                        difficult_text = doc.createTextNode('0')

                        name.appendChild(name_text)

                        lobject.appendChild(name)

                        pose.appendChild(pose_text)

                        lobject.appendChild(pose)

                        truncated.appendChild(truncated_text)

                        lobject.appendChild(truncated)

                        difficult.appendChild(difficult_text)

                        lobject.appendChild(difficult)

                        top, left, bottom, right = box

                        top = max(0, np.floor(top + 0.5).astype('int32'))

                        left = max(0, np.floor(left + 0.5).astype('int32'))

                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        bndbox = doc.createElement('bndbox')

                        xmin = doc.createElement('xmin')

                        xmin_text = doc.createTextNode(str(left))

                        ymin = doc.createElement('ymin')

                        ymin_text = doc.createTextNode(str(top))

                        xmax = doc.createElement('xmax')

                        xmax_text = doc.createTextNode(str(right))

                        ymax = doc.createElement('ymax')

                        ymax_text = doc.createTextNode(str(bottom))

                        xmin.appendChild(xmin_text)

                        bndbox.appendChild(xmin)

                        ymin.appendChild(ymin_text)

                        bndbox.appendChild(ymin)

                        xmax.appendChild(xmax_text)

                        bndbox.appendChild(xmax)

                        ymax.appendChild(ymax_text)

                        bndbox.appendChild(ymax)

                        lobject.appendChild(bndbox)

                        annotation.appendChild(lobject)
                fp = open('%s.xml' % lxml, 'w+')

                doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

                fp.close()

                end = timer()

                print(end - start)

            def detect_video(yolo, video_path, output_path=""):
                import cv2
                vid = cv2.VideoCapture('123.mp4')
                if not vid.isOpened():
                    raise IOError("Couldn't open webcam or video")
                video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
                video_fps = vid.get(cv2.CAP_PROP_FPS)
                video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                isOutput = True if output_path != "" else False
                if isOutput:
                    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
                    out = cv2.VideoWriter('11223344.mp4', video_FourCC, video_fps, video_size)
                accum_time = 0
                curr_fps = 0
                fps = "FPS: ??"
                prev_time = timer()
                while True:
                    return_value, frame = vid.read()
                    if return_value:
                        print('当前数据读入正确，此处放入正确代码！')
                        image = Image.fromarray(frame)
                        image = yolo.detect_image(image)
                        result = np.asarray(image)
                        curr_time = timer()
                        exec_time = curr_time - prev_time
                        prev_time = curr_time
                        accum_time = accum_time + exec_time
                        curr_fps = curr_fps + 1
                    else:
                        print('当前数据读入错误，跳出当前循环……')
                        break
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(255, 0, 0), thickness=2)
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", result)
                    if isOutput:
                        out.write(result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                yolo.close_session()

            def close_session(self):
                self.sess.close()

        # kaishi
        def detect_xml112(yolo):
            print(896)
            #name = fname1
            #print(name)
            #fnamee = name.split("/", )
            #print(908, len(fnamee), fnamee)
            #ggwewe = os.path.join(fnamee[len(fnamee) - 4], fnamee[len(fnamee) - 3], fnamee[len(fnamee) - 2])
            #print(910, ggwewe)
            # llist = get_all_file('VOCdevkit\VOC2007\JPEGImages12')
            llist = get_all_file(pic_path)
            # folder = r"VOCdevkit\VOC2007\JPEGImages1\xml56"
            folder = isExists
            num_count = 0

            for index in range(len(llist) - 1):
                img = llist[index]
                # lujing='VOCdevkit\VOC2007\xml28.ll'
                print(920, (img.split('/')[-1]))
                print(921, os.path.splitext(img.split('/')[-1])[0])
                print(img)  # 图片
                image = Image.open(img)
                print("916")
                #print(image)  # 3通道
                #lxml = os.path.join((pic_path+zaisheng_path),os.path.splitext(img.split('/')[-1])[0] )
                lxml = os.path.join((zaisheng_path),os.path.splitext(img.split('/')[-1])[0] )
                #gg1 = lxml.split("\\", -2)
                #print(928, gg1)
                #gg = os.path.join(gg1[0], gg1[1], gg1[2])
                #print(930, gg)
                #lxml = os.path.join(gg, folder, gg1[3])
                print(932, lxml)
                r_image = yolo.detect_xml(image, lxml)

                num_count += 1

                print('这是第%d张图' % num_count)

            yolo.close_session()

        allfile = []

        def get_all_file(rawdir):
            allfilelist = os.listdir(rawdir)
            for f in allfilelist:
                filepath = os.path.join(rawdir, f)
                if os.path.isdir(filepath):
                    get_all_file(filepath)
                allfile.append(filepath)
            return allfile

        FLAGS = None
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        '''
        Command line options
        '''
        parser.add_argument(
            '--model', type=str,
            help='path to model weight file, default ' + YOLO.get_defaults("model_path")
        )

        parser.add_argument(
            '--anchors', type=str,
            help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
        )

        parser.add_argument(
            '--classes', type=str,
            help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
        )

        parser.add_argument(
            '--gpu_num', type=int,
            help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
        )

        parser.add_argument(
            '--image', default=False, action="store_true",
            help='Image detection mode, will ignore all positional arguments'
        )
        '''
        Command line positional arguments -- for video detection mode
        '''
        parser.add_argument(
            "--input", nargs='?', type=str, required=False, default='./path2your_video',
            help="Video input path"
        )

        parser.add_argument(
            "--output", nargs='?', type=str, default="",
            help="[Optional] Video output path"
        )

        FLAGS = parser.parse_args()
        print(982)
        if True:

            """

            Image detection mode, disregard any remaining command line arguments

            """

            print("996Image detection mode")
            if "input" in FLAGS:
                print(" 999Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
            # 图片xml
            print("1000start!!!")
            detect_xml112(YOLO(**vars(FLAGS)))
            # 视频xml
            # detect_video1(YOLO(**vars(FLAGS)),'F:/新建文件夹/lunwen/123.mp4')
        # image_path = 'F:/tensor1.5.0/keras-yolo3-master/keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages1/frame_000000.jpg'
        # image = Image.open(fname)
        # print(image)
        # image=np.array(image)
        # print(image.shape)
        self.label_name5.setStyleSheet("QLabel{background:pink;}"
                                       "QLabel{color:rgb(100,100,100,960);font-size:36px;font-weight:bold;font-family:楷体;}"
                                       )

        self.label_name5 = QLabel('完事儿了昂', self)
        self.label_name5.setGeometry(10, 20, 480, 480)
        print(1011)
        self.edit.setText('已完成:' + str("gg"))

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()
        self.f = FirstUi()
        self.f.show()

class picture_num(QWidget):
    def __init__(self):
        super(picture_num, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640,520)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('图片→xml')
        #self.label_name1 = QLabel('哔哩哔哩大学', self)
        #self.label_name1.setGeometry(500, 20, 120, 35)

        #self.label_name2 = QLabel('知识学院', self)
        #self.label_name2.setGeometry(500, 60, 100, 35)

        #self.label_name3 = QLabel('野生技术协会', self)
        #self.label_name3.setGeometry(500, 100, 100, 35)

        #self.label_name4 = QLabel('南岛鹋', self)
        #self.label_name4.setGeometry(500, 140, 100, 35)

        self.label_name5 = QLabel('🐱🐟🐕🦈🐅🦁🐘', self)
        self.label_name5.setGeometry(10, 20, 480, 480)
        self.label_name5.setStyleSheet("QLabel{background:pink;}"
                                 "QLabel{color:rgb(100,100,100,960);font-size:50px;font-weight:bold;font-family:楷体;}"
                                 )
        self.label_name5.setAlignment(QtCore.Qt.AlignCenter)
        
        self.edit = QTextEdit(self)
        self.edit.setGeometry(500, 220, 100, 60)

        self.btn_select = QPushButton('选图',self)
        self.btn_select.setGeometry(500, 320, 100, 30)
        self.btn_select.clicked.connect(self.select_image)

        self.btn_dis = QPushButton('生成xml',self)
        self.btn_dis.setGeometry(500, 370, 100, 30)
        self.btn_dis.clicked.connect(self.on_btn_Recognize_Clicked)

        self.btn = QPushButton('返回',self)
        self.btn.setGeometry(500, 420, 100, 30)
        self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出',self)
        self.btn_exit.setGeometry(500, 470, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_image(self):
        print(1071)
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(1074,self.label_name5.width(),self.label_name5.height(),imgName,imgType)

        img = Image.open(imgName)

        img.save("xianshi.png")
        jpg = QtGui.QPixmap("xianshi.png").scaled(self.label_name5.width(), self.label_name5.height()//2)
        print(1080,jpg)
        self.label_name5.setPixmap(jpg)
        print(1082)
        fname = imgName
        print(fname)
    def on_btn_Recognize_Clicked(self):
        global fname
        class YOLO(object):
            _defaults = {
                "model_path": 'logs/003/trained_weights_final.h5',
                "anchors_path": 'model_data/yolo_anchors.txt',
                "classes_path": 'model_data/coco_classes.txt',
                "score": 0.2,  # 0.3
                "iou": 0.25,  # 0.45
                "model_image_size": (416, 416),
                "gpu_num": 1,
            }

            @classmethod
            def get_defaults(cls, n):
                if n in cls._defaults:
                    return cls._defaults[n]
                else:
                    return "Unrecognized attribute name '" + n + "'"

            def __init__(self, **kwargs):
                self.__dict__.update(self._defaults)  # set up default values
                self.__dict__.update(kwargs)  # and update with user overrides
                self.class_names = self._get_class()
                self.anchors = self._get_anchors()
                self.sess = K.get_session()
                self.boxes, self.scores, self.classes = self.generate()

            def _get_class(self):
                classes_path = os.path.expanduser(self.classes_path)
                with open(classes_path) as f:
                    class_names = f.readlines()
                class_names = [c.strip() for c in class_names]
                return class_names

            def _get_anchors(self):
                anchors_path = os.path.expanduser(self.anchors_path)
                with open(anchors_path) as f:
                    anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                return np.array(anchors).reshape(-1, 2)

            def generate(self):
                model_path = os.path.expanduser(self.model_path)
                assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

                # Load model, or construct model and load weights.
                num_anchors = len(self.anchors)
                num_classes = len(self.class_names)
                is_tiny_version = num_anchors == 6  # default setting
                try:
                    self.yolo_model = load_model(model_path, compile=False)
                except:
                    self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                        if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
                    self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
                else:
                    assert self.yolo_model.layers[-1].output_shape[-1] == \
                           num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                        'Mismatch between model and given anchor and class sizes'

                print('{} model, anchors, and classes loaded.'.format(model_path))

                # Generate colors for drawing bounding boxes.
                hsv_tuples = [(x / len(self.class_names), 1., 1.)
                              for x in range(len(self.class_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)  # Fixed seed for consistent colors across runs.
                np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
                np.random.seed(None)  # Reset seed to default.

                # Generate output tensor targets for filtered bounding boxes.
                self.input_image_shape = K.placeholder(shape=(2,))
                if self.gpu_num >= 2:
                    self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
                boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                   len(self.class_names), self.input_image_shape,
                                                   score_threshold=self.score, iou_threshold=self.iou)
                return boxes, scores, classes

            def detect_xml(self, image, lxml):
                start = timer()
                print(1175)
                if self.model_image_size != (None, None):

                    assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'

                    assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'

                    boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))

                else:

                    new_image_size = (image.width - (image.width % 32),

                                      image.height - (image.height % 32))

                    boxed_image = letterbox_image(image, new_image_size)

                image_data = np.array(boxed_image, dtype='float32')

                print(image_data.shape)

                image_data /= 255.

                image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

                out_boxes, out_scores, out_classes = self.sess.run(

                    [self.boxes, self.scores, self.classes],

                    feed_dict={

                        self.yolo_model.input: image_data,

                        self.input_image_shape: [image.size[1], image.size[0]],

                        K.learning_phase(): 0

                    })
                print('142', out_boxes, out_scores, out_classes)  # print("s447",label, (left, top), (right, bottom))
                doc = xml.dom.minidom.Document()

                annotation = doc.createElement('annotation')

                doc.appendChild(annotation)

                folder = doc.createElement('folder')

                folder_text = doc.createTextNode('blank')

                folder.appendChild(folder_text)

                annotation.appendChild(folder)

                filename = doc.createElement('filename')

                filename_text = doc.createTextNode('blank')

                filename.appendChild(filename_text)

                annotation.appendChild(filename)

                path = doc.createElement('path')

                path_text = doc.createTextNode('blank')

                path.appendChild(path_text)

                annotation.appendChild(path)

                source = doc.createElement('source')

                databass = doc.createElement('databass')

                databass_text = doc.createTextNode('Unknown')

                source.appendChild(databass)

                databass.appendChild(databass_text)

                annotation.appendChild(source)

                size = doc.createElement('size')

                width = doc.createElement('width')

                width_text = doc.createTextNode(str(image.width))

                height = doc.createElement('height')

                height_text = doc.createTextNode(str(image.height))

                depth = doc.createElement('depth')

                depth_text = doc.createTextNode('3')

                size.appendChild(width)

                width.appendChild(width_text)

                size.appendChild(height)

                height.appendChild(height_text)

                size.appendChild(depth)

                depth.appendChild(depth_text)

                annotation.appendChild(size)

                segmented = doc.createElement('segmented')

                segmented_text = doc.createTextNode('0')

                segmented.appendChild(segmented_text)

                annotation.appendChild(segmented)

                for i, c in reversed(list(enumerate(out_classes))):

                    predicted_class = self.class_names[c]

                    if str(predicted_class) == 'screen':
                        box = out_boxes[i]

                        lobject = doc.createElement('object')

                        name = doc.createElement('name')

                        name_text = doc.createTextNode(str(predicted_class))

                        pose = doc.createElement('pose')

                        pose_text = doc.createTextNode('Unspecified')

                        truncated = doc.createElement('truncated')

                        truncated_text = doc.createTextNode('0')

                        difficult = doc.createElement('difficult')

                        difficult_text = doc.createTextNode('0')

                        name.appendChild(name_text)

                        lobject.appendChild(name)

                        pose.appendChild(pose_text)

                        lobject.appendChild(pose)

                        truncated.appendChild(truncated_text)

                        lobject.appendChild(truncated)

                        difficult.appendChild(difficult_text)

                        lobject.appendChild(difficult)

                        top, left, bottom, right = box

                        top = max(0, np.floor(top + 0.5).astype('int32'))

                        left = max(0, np.floor(left + 0.5).astype('int32'))

                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        bndbox = doc.createElement('bndbox')

                        xmin = doc.createElement('xmin')

                        xmin_text = doc.createTextNode(str(left))

                        ymin = doc.createElement('ymin')

                        ymin_text = doc.createTextNode(str(top))

                        xmax = doc.createElement('xmax')

                        xmax_text = doc.createTextNode(str(right))

                        ymax = doc.createElement('ymax')

                        ymax_text = doc.createTextNode(str(bottom))

                        xmin.appendChild(xmin_text)

                        bndbox.appendChild(xmin)

                        ymin.appendChild(ymin_text)

                        bndbox.appendChild(ymin)

                        xmax.appendChild(xmax_text)

                        bndbox.appendChild(xmax)

                        ymax.appendChild(ymax_text)

                        bndbox.appendChild(ymax)

                        lobject.appendChild(bndbox)

                        annotation.appendChild(lobject)
                    if str(predicted_class) == 'icon':
                        box = out_boxes[i]

                        lobject = doc.createElement('object')

                        name = doc.createElement('name')

                        name_text = doc.createTextNode(str(predicted_class))

                        pose = doc.createElement('pose')

                        pose_text = doc.createTextNode('Unspecified')

                        truncated = doc.createElement('truncated')

                        truncated_text = doc.createTextNode('0')

                        difficult = doc.createElement('difficult')

                        difficult_text = doc.createTextNode('0')

                        name.appendChild(name_text)

                        lobject.appendChild(name)

                        pose.appendChild(pose_text)

                        lobject.appendChild(pose)

                        truncated.appendChild(truncated_text)

                        lobject.appendChild(truncated)

                        difficult.appendChild(difficult_text)

                        lobject.appendChild(difficult)

                        top, left, bottom, right = box

                        top = max(0, np.floor(top + 0.5).astype('int32'))

                        left = max(0, np.floor(left + 0.5).astype('int32'))

                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        bndbox = doc.createElement('bndbox')

                        xmin = doc.createElement('xmin')

                        xmin_text = doc.createTextNode(str(left))

                        ymin = doc.createElement('ymin')

                        ymin_text = doc.createTextNode(str(top))

                        xmax = doc.createElement('xmax')

                        xmax_text = doc.createTextNode(str(right))

                        ymax = doc.createElement('ymax')

                        ymax_text = doc.createTextNode(str(bottom))

                        xmin.appendChild(xmin_text)

                        bndbox.appendChild(xmin)

                        ymin.appendChild(ymin_text)

                        bndbox.appendChild(ymin)

                        xmax.appendChild(xmax_text)

                        bndbox.appendChild(xmax)

                        ymax.appendChild(ymax_text)

                        bndbox.appendChild(ymax)

                        lobject.appendChild(bndbox)

                        annotation.appendChild(lobject)
                fp = open('%s.xml' % lxml, 'w+')

                doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

                fp.close()

                end = timer()

                print(end - start)



            def detect_video(yolo, video_path, output_path=""):
                import cv2
                vid = cv2.VideoCapture('F:/新建文件夹/lunwen/123.mp4')
                if not vid.isOpened():
                    raise IOError("Couldn't open webcam or video")
                video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
                video_fps = vid.get(cv2.CAP_PROP_FPS)
                video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                isOutput = True if output_path != "" else False
                if isOutput:
                    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
                    out = cv2.VideoWriter('F:/新建文件夹/lunwen/11223344.mp4', video_FourCC, video_fps, video_size)
                accum_time = 0
                curr_fps = 0
                fps = "FPS: ??"
                prev_time = timer()
                while True:
                    return_value, frame = vid.read()
                    if return_value:
                        print('当前数据读入正确，此处放入正确代码！')
                        image = Image.fromarray(frame)
                        image = yolo.detect_image(image)
                        result = np.asarray(image)
                        curr_time = timer()
                        exec_time = curr_time - prev_time
                        prev_time = curr_time
                        accum_time = accum_time + exec_time
                        curr_fps = curr_fps + 1
                    else:
                        print('当前数据读入错误，跳出当前循环……')
                        break
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(255, 0, 0), thickness=2)
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", result)
                    if isOutput:
                        out.write(result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                yolo.close_session()

            def close_session(self):
                self.sess.close()
        #kaishi
        def detect_xml112(yolo):
            print(1536)
            name=fname
            print(name)
            fnamee = name.split("/",)
            print(1540,len(fnamee),fnamee)
            ggwewe = os.path.join(fnamee[len(fnamee)-4], fnamee[len(fnamee)-3], fnamee[len(fnamee)-2])
            print(1542,ggwewe)
            #llist = get_all_file('VOCdevkit\VOC2007\JPEGImages12')
            llist = get_all_file(ggwewe)
            # folder = r"VOCdevkit\VOC2007\JPEGImages1\xml56"
            folder = "xmlkong"
            num_count = 0

            for index in range(len(llist) - 1):
                img = llist[index]
                # lujing='VOCdevkit\VOC2007\xml28.ll'
                print(33, (img.split('/')[-1]))
                print(35, os.path.splitext(img.split('/')[-1])[0])
                print(img)  # 图片
                image = Image.open(img)
                print("38")
                print(image)  # 3通道
                lxml = os.path.join(os.path.splitext(img.split('/')[-1])[0], )
                gg1 = lxml.split("\\", -2)
                print(42, gg1)
                gg = os.path.join(gg1[0], gg1[1], gg1[2])
                print(44, gg)
                lxml = os.path.join(gg, folder, gg1[3])
                print(46, lxml)
                r_image = yolo.detect_xml(image, lxml)

                num_count += 1

                print('这是第%d张图' % num_count)

            yolo.close_session()
        allfile = []
        def get_all_file(rawdir):
            allfilelist = os.listdir(rawdir)
            for f in allfilelist:
                filepath = os.path.join(rawdir, f)
                if os.path.isdir(filepath):
                    get_all_file(filepath)
                allfile.append(filepath)
            return allfile

        FLAGS = None
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        '''
        Command line options
        '''
        parser.add_argument(
            '--model', type=str,
            help='path to model weight file, default ' + YOLO.get_defaults("model_path")
        )

        parser.add_argument(
            '--anchors', type=str,
            help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
        )

        parser.add_argument(
            '--classes', type=str,
            help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
        )

        parser.add_argument(
            '--gpu_num', type=int,
            help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
        )

        parser.add_argument(
            '--image', default=False, action="store_true",
            help='Image detection mode, will ignore all positional arguments'
        )
        '''
        Command line positional arguments -- for video detection mode
        '''
        parser.add_argument(
            "--input", nargs='?', type=str, required=False, default='./path2your_video',
            help="Video input path"
        )

        parser.add_argument(
            "--output", nargs='?', type=str, default="",
            help="[Optional] Video output path"
        )

        FLAGS = parser.parse_args()
        print(943)
        if True:

            """

            Image detection mode, disregard any remaining command line arguments

            """

            print("1632Image detection mode")

            if "input" in FLAGS:
                print(" 1635--Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
            # 图片xml
            print("1637start!!!")
            detect_xml112(YOLO(**vars(FLAGS)))
            # 视频xml
            # detect_video1(YOLO(**vars(FLAGS)),'F:/新建文件夹/lunwen/123.mp4')
        #image_path = 'F:/tensor1.5.0/keras-yolo3-master/keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages1/frame_000000.jpg'
        #image = Image.open(fname)
        # print(image)
        # image=np.array(image)
        # print(image.shape)
        self.label_name5.setStyleSheet("QLabel{background:pink;}"
                                       "QLabel{color:rgb(100,100,100,960);font-size:36px;font-weight:bold;font-family:楷体;}"
                                       )

        self.label_name5 = QLabel('完事儿了昂', self)
        self.label_name5.setGeometry(10, 20, 480, 480)
        print(575)
        self.edit.setText('已完成:' + str("gg"))
        
    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()
        self.f = FirstUi()
        self.f.show()


def main():
    app = QApplication(sys.argv)
    w = FirstUi()  # 将第一和窗口换个名字
    w.show()  # 将第一和窗口换个名字显示出来
    sys.exit(app.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）


if __name__ == '__main__':  # 只有在本py文件中才能用，被调用就不执行
    main()
