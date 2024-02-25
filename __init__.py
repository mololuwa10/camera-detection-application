import cv2
import numpy as np

class objectDetection:
    def __init__(self, weights_path='yolov4.weights', cfg_path='yolov4.cfg'):
        self.nmsThreshold = 0.4
        self.confidenceThreshold = 0.5
        self.image_size = 608

        network = cv2.dnn.readNet(weights_path, cfg_path)

        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(network)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80,3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="classes.txt"):
        with open(classes_path, 'r') as file_object:
            for class_names in file_object.readlines():
                class_name = class_names.strip()
                self.classes.append(class_name)

            self.colors = np.random.uniform(0, 255, size=(80, 3))
            return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold = self.nmsThreshold, confidenceThreshold = self.confidenceThreshold)