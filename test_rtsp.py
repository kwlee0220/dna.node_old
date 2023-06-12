
from time import sleep
from contextlib import closing

import cv2
import numpy as np

from dna import Box
from dna.camera.opencv_camera import OpenCvCamera, OpenCvVideFile
from dna.support.rectangle_drawer import RectangleDrawer

import os

# url = "data/etri/etri_041.mp4"
url = "rtsp://localhost:8554/visual"
# url = "rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/6/media.smp"
# url = "rtsp://admin:dnabased24@129.254.82.33:558/PlaybackChannel/3/media.smp/start=20220502T085000&end=20220502T090000"
# url = "rtsp://admin:dnabased24@129.254.82.33:558/PlaybackChannel/3/media.smp/start=20220502T085000"

import subprocess

proc = subprocess.Popen(["C:\\local\\ffmpeg\\bin\\ffmpeg", "-re",
                "-rtsp_transport", "tcp", "-i",
                # "rtsp://admin:dnabased24@129.254.82.33:558/PlaybackChannel/3/media.smp/start=20220502T085000&end=20220502T090000",
                "rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/6/media.smp",
                "-rtsp_transport", "tcp", "-c:v", "copy",
                "-f", "rtsp", "rtsp://localhost:8554/visual"],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
proc.communicate()

cv2.waitKey(5000)
# # vcap = cv2.VideoCapture("rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/6/media.smp")
vcap = cv2.VideoCapture(url)
while (1):
    ret, frame = vcap.read()
    if ( ret ):
        cv2.imshow('VIDEO', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        cv2.waitKey(1000)
        vcap = cv2.VideoCapture(url)

vcap.release()
proc.kill()

cv2.destroyAllWindows()    