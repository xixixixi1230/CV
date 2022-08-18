import os
import cv2
from base_camera import BaseCamera

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('无法打开摄像头')
        for i in range(10):
            _, img = camera.read()
            if i==9:
                cv2.imwrite('static/img/video/catch.jpg', img)
        while True:
            _, img = camera.read()

            # encode 成为 jpeg image 然后 return it
            yield cv2.imencode('.jpg', img)[1].tobytes()

