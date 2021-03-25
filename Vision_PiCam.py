#!/usr/bin/env python3
# import the necessary packages

import cv2
import numpy as np
import math

import datetime
import time

from pathlib import Path
import pickle

from networktables import NetworkTables
import cscore as cs
import logging

from FrameReaderPi import CameraVideoStream

from threading import Thread

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

tvecXSaved = np.array([0])
tvecYSaved = np.array([0])
tvecZSaved = np.array([0])

# Calibration for camera is 1080p
mtx = None
dist = None

# 640 by 360
xFactor = 3
yFactor = 3

# init camera server and network tables
logging.basicConfig(level=logging.DEBUG)
ip = "10.28.34.2"
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

cs = cs.CameraServer.getInstance()

outputStream = cs.putVideo("vision", 640, 360)

# # Input
# # Distance dataset
# distance = np.array([[2.0], [2.25], [2.5], [2.75], [3.0], [3.25], [3.5], [3.7], 
#                      [4.0], [4.3], [4.6], [4.8], [5.0], [5.2], [5.5], [5.75], 
#                      [6.0], [6.3], [6.5], [6.8], [7.0], [7.3], [7.5], [7.8], 
#                      [8.0], [8.2], [8.5], [9.0], [9.6], [10.0], [10.5], [10.8]])

# # Output
# # Shooter speed (RPM), Hood angle (deg)
# output = np.array([[3000, 80], [3000, 75], [3000, 70], [3000, 65], [3300, 60], [3300, 57], [3300, 56], [3300, 55], 
#                    [3300, 54], [3300, 53], [3600, 53], [3600, 52], [3600, 51], [3700, 51], [3700, 50], [3800, 50], 
#                    [3800, 49], [3900, 49], [3900, 48], [3900, 47], [4100, 47], [4200, 47], [4200, 47], [4400, 47], 
#                    [4400, 46], [4500, 46], [4600, 46], [4700, 45], [4800, 45], [4900, 45], [5000, 45], [5600, 30]])

# # Verification set
# distance2 = np.array([[3.0], [3.2], [3.4], [3.6], [3.8], [4.0], [4.2], [4.4], [4.6], 
#                       [4.8], [5.0], [5.2], [5.4], [5.6], [5.8], [6.0], [6.2], [6.4], 
#                       [6.6], [6.8], [7.0], [7.2], [7.4], [7.6], [7.8], [8.0], [8.2], 
#                       [8.4], [8.6], [8.8], [9.0], [9.2], [9.4], [9.6], [9.8], [10.0]])

# knn = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', 
#                           leaf_size=30, p=2, metric='minkowski', metric_params=None, 
#                           n_jobs=None)
# regr = MultiOutputRegressor(knn)

# regr.fit(distance, output)

# filename = 'shooterModel1.sav'
# pickle.dump(regr, open(filename, 'wb'))

# model for shooter rpm and angle
filename = 'shooterModel1.sav'
model = pickle.load(open(filename, mode='rb'))

# calibrated for 1080p
distortion_correction_file = Path("distortion_correction_pickle_ir_wide_angle_1080_2.p")
# check if we already created the calibration file with coefficients
if distortion_correction_file.is_file():
    # load the coefficients to undistort the camera image
    with open('distortion_correction_pickle_ir_wide_angle_1080_2.p', mode='rb') as f:
        calibration_file = pickle.load(f)
        mtx, dist = calibration_file['mtx'], calibration_file['dist']
else:
    print('Calibration does not exist.')

print("mtx: ", mtx)
print("dist: ", dist)

# Scale camera matrix to allow for different frame sizes
#fx
mtx[0,0] = mtx[0,0] / xFactor
#cx
mtx[0,2] = mtx[0,2] / xFactor
#fy
mtx[1,1] = mtx[1,1] / yFactor
#cy
mtx[1,2] = mtx[1,2] / yFactor

print("mtx:", mtx)

# Inch to Meter conversion
inToMConversion = 0.0254
# Real World points of vision target in inches
objectPoints = np.array([[-19.625 * inToMConversion ,0,0], [19.625 * inToMConversion,0,0], [(19.625-9.8051325845) * inToMConversion,-17 * inToMConversion,0], [(-19.625+9.8051325845) * inToMConversion,-17 * inToMConversion,0]], dtype=np.float32)
# Virtual World points of trihedron to show target pose
trihedron = np.array([[0,0,0],[12 * inToMConversion,0,0],[0,12 * inToMConversion,0],[0,0,12 * inToMConversion]], dtype=np.float32)

# Contour filtering constants 
# Expected = 2.3088235294
aspectRatioMin = 1.0
aspectRatioMax = 3.0

# Expected = 0.2206857965
solidityMin = 0.1
solidityMax = 0.5

polySides = 4

#Class to examine Frames per second of camera stream.
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

cap = CameraVideoStream("/dev/video0", True).start()

# Gets the contours of the target
def getContours(frame):
    # Mask the target
    blur = cv2.blur(frame,(3,3), -1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, np.array([50,50,40]), np.array([93,255,255]))
    mask = cv2.inRange(hsv, np.array([0,0,100]), np.array([255,80,255]))

    # Find contours of the target
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blur, contours, -1, (0,255,255), 3)

    #outputStream.putFrame(blur)

    return blur, contours

# Filter the contours
def filterContours(frame, contours):
    if len(contours) > 0:
        #Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in cntsSorted:

            #cnt = cntsSorted[0]

            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)

            # Aspect ratio of the target
            aspectRatio = w/h
            # Solidity of the target
            solidity = 0
            if hullArea!=0:
                solidity = cntArea/hullArea

            # Calculate the epsilon for the polygon
            # epsilon = 0.045*cv2.arcLength(hull, closed=True)
            # epsilon = 0.015*cv2.arcLength(hull, closed=True)
            epsilon = 0.04*cv2.arcLength(hull, closed=True)
            # Approx a polygon fit to the convex hull
            approx = cv2.approxPolyDP(hull, epsilon, True)

            cv2.drawContours(frame, [approx], -1, (255, 255, 0), 3)

            print("aspectRatio", aspectRatio)
            print("solidity", solidity)
            print("hullSides", len(approx))

            sd.putNumber("aspectRatio", aspectRatio)
            sd.putNumber("solidity", solidity)
            sd.putNumber("hullSides", len(approx))

            # Check if contour matches criteria
            if (aspectRatio>=aspectRatioMin and aspectRatio<=aspectRatioMax) and (solidity>=solidityMin and solidity<=solidityMax) and (len(approx)==polySides):
                print("TARGET DETECTED")

                return True, approx, frame

    return False, None, frame

def solvePNP(frame, approx):
    # Sort the corners
    approxSortedY = sorted(approx, key=lambda k: k[0][1])
    topCorners = [approxSortedY[0], approxSortedY[1]]
    bottomCorners = [approxSortedY[2], approxSortedY[3]]
    bottomCornersSortedX = sorted(bottomCorners, key=lambda k: k[0][0])
    topCornersSortedX = sorted(topCorners, key=lambda k: k[0][0])

    # Corners, top left, top right, bottom right, bottom left
    corners = np.array([topCornersSortedX[0], topCornersSortedX[1], bottomCornersSortedX[1], bottomCornersSortedX[0]],dtype=np.float32)

    cv2.line(frame, (corners[0][0][0],corners[0][0][1]), (corners[0][0][0],corners[0][0][1]), (255,0,0), 5)
    cv2.line(frame, (corners[1][0][0],corners[1][0][1]), (corners[1][0][0],corners[1][0][1]), (0,255,0), 5)
    cv2.line(frame, (corners[2][0][0],corners[2][0][1]), (corners[2][0][0],corners[2][0][1]), (0,0,255), 5)
    cv2.line(frame, (corners[3][0][0],corners[3][0][1]), (corners[3][0][0],corners[3][0][1]), (0,255,255), 5)

    # Get the pose of the vision taraget
    retval, rvec, tvec = cv2.solvePnP(objectPoints, corners, mtx, dist, cv2.SOLVEPNP_P3P)

    trihedronPoints, _ = cv2.projectPoints(trihedron, rvec, tvec, mtx, dist)

    cv2.putText(frame, "tvec x: " + str("%.2f" %(tvec[0][0])), (40, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5,
        (255, 255, 255))
    cv2.putText(frame, "tvec y: " + str("%.2f" %(tvec[1][0])), (40, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5,
        (255, 255, 255))
    cv2.putText(frame, "tvec z: " + str("%.2f" %(tvec[2][0])), (40, 210), cv2.FONT_HERSHEY_COMPLEX, 0.5,
        (255, 255, 255))

    if ((trihedronPoints[0][0][0] <= 10000) and (trihedronPoints[0][0][0] >= -10000)) and ((trihedronPoints[0][0][1] <= 10000) and (trihedronPoints[0][0][1] >= -10000)) and ((trihedronPoints[1][0][0] <= 10000) and (trihedronPoints[1][0][0] >= -10000)) and ((trihedronPoints[1][0][1] <= 10000) and (trihedronPoints[1][0][1] >= -10000)) and ((trihedronPoints[2][0][0] <= 10000) and (trihedronPoints[2][0][0] >= -10000)) and ((trihedronPoints[2][0][1] <= 10000) and (trihedronPoints[2][0][1] >= -10000)) and ((trihedronPoints[3][0][0] <= 10000) and (trihedronPoints[3][0][0] >= -10000)) and ((trihedronPoints[3][0][1] <= 10000) and (trihedronPoints[3][0][1] >= -10000)):
        cv2.line(frame, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[1][0][0]), int(trihedronPoints[1][0][1])), (255,0,0), 3)
        cv2.line(frame, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[2][0][0]), int(trihedronPoints[2][0][1])), (0,255,0), 3)
        cv2.line(frame, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[3][0][0]), int(trihedronPoints[3][0][1])), (0,0,255), 3)

    processTvec(tvec)

# Constants
# Projectile parameters
ballDiameter = 0.1778
ballRadius = ballDiameter/2
# Drag coefficient
Cd = 1.6
# Spin coefficient
Cs = 0.1
# Density of medium (kg/m^3)
p = 1.225 
# Projected area (m^2)
A = math.pi*ballRadius**2 
# Proportionality constant (kg)
Ks = 0.5*Cs*p*A*ballRadius
# Mass of object (kg)
m = 0.146
# Acceleration due to gravity (m/s^2)
g = 9.8
# Time step
t = 0.01

# Constant height to the target and the shooter height
targetHeight = 2.49555
shooterHeight = 0.4572
height = targetHeight-shooterHeight

def processTvec(tvec):
    x = tvec[0][0]
    y = tvec[1][0]
    z = tvec[2][0]

    # Rotate tvec to be parallel with the ground
    camAngleOfElevation = 15 * (math.pi / 180)
    camHypot = math.hypot(y, z)
    camTheta = math.atan(-y / z)
    y = camHypot * math.sin(camTheta + camAngleOfElevation)
    z = camHypot * math.cos(camTheta + camAngleOfElevation)

    # Translate tvec to the center of the turret turning location
    # xDelta = -0.15
    # yDelta = 0
    # zDelta = 0.127
    xDelta = 0.0
    yDelta = 0.0
    zDelta = 0.0
    x1 = x + xDelta
    y1 = y + yDelta
    z1 = z + zDelta

    # Get the yaw to the target
    yaw = math.atan(x1 / z1)

    # Translate tvec to the center of the flywheel
    xDelta2 = -0.15
    yDelta2 = 0.1
    zDelta2 = 0.094
    x2 = x + xDelta2
    y2 = y + yDelta2
    z2 = z + zDelta2

    # Get the distance to the target
    distance = math.hypot(x1, z1)

    # Get the desired hood angle
    pitchToTarget = math.atan((height) / distance)
    #hoodAngle = pitchToTarget
    #hoodAngle = ((math.pi / 2) + pitchToTarget) / 2
    #hoodAngle = pitchToTarget + (math.pi / 2 - pitchToTarget) / 3

    # Get ideal init v
    # a = distance ** 2 * 9.8
    # b = distance * math.sin(2 * hoodAngle) - 2 * height * math.cos(hoodAngle) ** 2
    # initV = math.sqrt(a / b)

    # # Get Vx, Vy, and W
    # Vx, Vy, w = getVxVyw(initV , hoodAngle)
    # # Get Ax and Ay
    # Ax, Ay = getAxAy(Cd, Ks, w, p, A, m, g, Vx, Vy)
    # correctedInitV = getTargetVelocity(initV, hoodAngle, distance)

    # Get the speed of the shooter wheel in radians per second
    # shooterRadius = 0.0762
    # tangentialV = initV * 2
    # radiansPerSec = tangentialV / shooterRadius

    # Use the model to get the shooter rpm and angle
    output = model.predict([[distance]])
    print(output)

    sd.putNumber("targetRPM", output[0][0])
    sd.putNumber("hoodAngle (deg)", output[0][1])

    # print(correctedInitV, "m/s")

    sd.putNumber("turretYawError", yaw)
    # sd.putNumber("targetHoodAngle", hoodAngle)
    sd.putNumber("distance", distance)
    # sd.putNumber("shooterV (rads/sec)", radiansPerSec)
    sd.putNumber("pitch (deg)", pitchToTarget * (180 / math.pi))

def getVxVyw(initialVelocity, shootingAngle):
    Vx = initialVelocity * math.cos(shootingAngle)
    Vy = initialVelocity * math.sin(shootingAngle)
    # For a flywheel/hood shooter
    tangentialVelocity = initialVelocity * 2
    w = tangentialVelocity / ballRadius
#     print(Vx, "m/s")
#     print(Vy, "m/s")
#     print(w, "radians/s")
    return Vx, Vy, w

def getAxAy(Cd, Ks, w, p, A, m, g, Vx, Vy):
    Ax = -0.5 * Cd * p * A * Vx * math.sqrt(Vx ** 2 + Vy ** 2) * (1 / m) - Ks * w * Vy * (1 / m)
    Ay = -0.5 * Cd * p * A * Vy * math.sqrt(Vx ** 2 + Vy ** 2) * (1 / m) - g + Ks * w * Vx * (1 / m)
#     print(Ax, "m/s^2")
#     print(Ay, "m/s^2")
    return Ax, Ay

def getHeightAtTarget(Ax, Ay, Vx, Vy, x, y, t, w, distanceToTarget, targetHeight):
    currentAx = Ax
    currentAy = Ay
    currentVx = Vx
    currentVy = Vy
    currentX = x
    currentY = y
    
    while currentX < distanceToTarget:
        currentX = currentVx * t + currentX
        currentY = currentVy * t + currentY
        currentVx = currentAx * t + currentVx
        currentVy = currentAy * t + currentVy
        currentAx, currentAy = getAxAy(Cd, Ks, w, p, A, m, g, currentVx, currentVy)
        
#     print(y)
    return currentX, currentY

def getTargetVelocity(initV, hoodAngle, distance):
    initialVelocity = initV
    Vx, Vy, w = getVxVyw(initialVelocity , hoodAngle)
    Ax, Ay = getAxAy(Cd, Ks, w, p, A, m, g, Vx, Vy)
    x, y = getHeightAtTarget(Ax, Ay, Vx, Vy, 0, shooterHeight, t, w, distance, targetHeight)

    if y > targetHeight:
        while y > targetHeight:
            initialVelocity -= 0.1
            Vx, Vy, w = getVxVyw(initialVelocity , hoodAngle)
            Ax, Ay = getAxAy(Cd, Ks, w, p, A, m, g, Vx, Vy)
            x, y = getHeightAtTarget(Ax, Ay, Vx, Vy, 0, shooterHeight, t, w, distance, targetHeight)
    if y < targetHeight:
        while y < targetHeight:
            initialVelocity += 0.1
            Vx, Vy, w = getVxVyw(initialVelocity , hoodAngle)
            Ax, Ay = getAxAy(Cd, Ks, w, p, A, m, g, Vx, Vy)
            x, y = getHeightAtTarget(Ax, Ay, Vx, Vy, 0, shooterHeight, t, w, distance, targetHeight)

    return initialVelocity

#fps = FPS().start()

while(True):

    frame, frameAquiredTime = cap.read()
    frame, contours = getContours(frame)
    retVal, approx, frame = filterContours(frame, contours)
    if retVal:
        sd.putBoolean("Target Detected?", True)
        solvePNP(frame, approx)
    else:
        sd.putBoolean("Target Detected?", False)
    outputStream.putFrame(frame)
    #Thread(target=findTape(contours, blur, (image_width/2)-0.5, (image_height/2)-0.5))

cap.release()