#!/usr/bin/env python3

# import the necessary packages
import pickle
from networktables import NetworkTables
import math

# Use UDP?
useUDP = False

if useUDP:
    print("Using UDP")
else:
    print("Using NetworkTables")

# model for shooter rpm and angle
filename = 'shooterModel.sav'
model = pickle.load(open(filename, mode='rb'))

# setting up network tables
ip = "10.28.34.2"
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

# field constants
power_port_height = 5
camera_height = 1
net_height = power_port_height - camera_height

limelight_angle = 10

def calculateRange(targetHeight, cameraAngle, vertOffset):
    netOffset = cameraAngle + vertOffset
    netOffsetRad = math.radians(netOffset)

    targetRange = targetHeight / math.tan(netOffsetRad)

    return targetRange

# TODO add udp to send shooter ouput
while True:
    target_detected = sd.getBoolean("tv", False)

    if target_detected:
        targetRange = calculateRange(net_height, limelight_angle, sd.getNumber("ty", 0))
        output = model.predict([[targetRange]])

        if not useUDP:
            sd.putNumber("targetRPM", output[0][0])
            sd.putNumber("targetAngle", output[0][1])
            NetworkTables.flush()

        print("RPM", output[0][0])
        print("Angle", output[0][1])
    else:
        print("No target detected")