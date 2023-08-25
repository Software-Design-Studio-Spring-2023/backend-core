import cv2
from imageai.Detection import VideoObjectDetection
import os

class ObjectDetection() :

    def __init__(self, client_url,model_path=os.path.join(os.getcwd() + "yolov3.pt"), output_path= os.path.join(os.getcwd(), "camera_detected_video"), frames_per_second=10, log_progress=True) :
        self.model_path = model_path
        self.client_url = client_url
        self.output_path = output_path
        self.frames_per_second = frames_per_second
        self.log_progress = log_progress

    def detect(self) :
        videoCapture = cv2.VideoCapture(self.client_url)
        detector = VideoObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(self.model_path)
        detector.loadModel()

        video_path = detector.detectObjectsFromVideo(camera_input=videoCapture,
                                                     frames_per_second=self.frames_per_second,
                                                     log_progress=self.log_progress)
        return video_path
    
