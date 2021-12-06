import donkeycar as dk
from donkeycar.vehicle import Vehicle
from donkeycar.parts.segmentation import PathSeg
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.robohat import RoboHATDriver


IMAGE_H=480
IMAGE_W=640
CAMERA_FRAMERATE=30
CAMERA_VFLIP= False
CAMERA_HFLIP= False

cfg = dk.load_config(myconfig="myconfig.py")

V=Vehicle()

cam = PiCamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=3, framerate=CAMERA_FRAMERATE, vflip=CAMERA_VFLIP, hflip=CAMERA_HFLIP)

V.add(cam,outputs=['cam/image_array'], threaded=False)	
				

# add your class that you have created in parts        
V.add(PathSeg(), inputs=['cam/image_array'], outputs=['pilot_angle','pilot_throttle'])


#Drive train setup
V.add(RoboHATDriver(cfg), inputs=["pilot_angle", "pilot_throttle"])


V.start()