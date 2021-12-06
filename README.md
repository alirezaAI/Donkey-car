# Donkey-car
Donkey car self-driving agent using Path Segmentation with deeplabv3-mobilenetv2 model

I have used a pre-existing segmentation model to isolate the route from rest of the scene and then use vertical sliding window search to estimate how the route 
changes and based on the changes calculate steering angle for the car to steer while the throttle is constant for a short period of time (0.6 seconds)
