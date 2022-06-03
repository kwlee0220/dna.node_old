
from contextlib import closing

from dna.camera.opencv_camera import OpenCvVideFile

cam = OpenCvVideFile("data/etri/etri_051.mp4")
with closing(cam.open()) as cap:
    while True:
        frame = cap()
        print(frame)