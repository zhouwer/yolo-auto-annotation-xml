# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

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

class YOLO(object):
    _defaults = {
        "model_path": 'logs/003/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.2,           #0.3
        "iou" : 0.25,             #0.45
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
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
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
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
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
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
        print('142',out_boxes, out_scores, out_classes)  #print("s447",label, (left, top), (right, bottom))
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
    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
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

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

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
        return image

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

def detect_video1(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture('F:/新建文件夹/lunwen/123.mp4')
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
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
        folder = ''
        if return_value:
            print('当前数据读入正确，此处放入正确代码！')
            print("540",frame.shape)  #矩阵,shape==3
            image = Image.fromarray(frame)
            print("542",image)  #3通道
            #lxml = os.path.join(folder, os.path.splitext(frame.split('/')[-1])[0])
            #image = yolo.detect_xml(image,lxml)    #yolo.detect_image
            image = yolo.detect_image
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
    yolo.close_session()       #未调用





if __name__ == '__main__':
    image_path = 'F:/tensor1.5.0/keras-yolo3-master/keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages1/frame_000007.jpg'
    image = Image.open(image_path)
    yolo = YOLO()
    r_image = yolo.detect_image(image=image)
    r_image.show()