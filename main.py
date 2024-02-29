import torch
import cv2
import os
from sympy import S, Interval
from shapely import Point, Polygon
import numpy as np
from ultralytics import YOLO
import ultralytics
from util_classes import Lane, Vehicle


def show_image(image, window_name='image', timeout=0):
    """
    :param timeout. How many seconds to wait untill it close the window.
    """
    cv2.imshow(window_name, cv2.resize(image, None, fx=0.6, fy=0.6))
    cv2.waitKey(timeout)
    cv2.destroyAllWindows()

def getLanes():

    lanes =  []

    lane1 = Lane([(260,410),(65,175), (90, 100), (410, 390)])
    lane2 = Lane([(410, 390),(90,100),(100,53),(530,380)])
    lane3 = Lane([(530,380),(50,10), (80,2),(640, 370)])

    lane4 = Lane([(1144, 375), (1920, 294),(1917, 320),(1190, 395)])
    lane5 = Lane([(1190, 395), (1917, 320), (1920, 345), (1245, 417)])
    lane6 = Lane([(1245, 417), (1920, 345), (1920, 375), (1297, 442)])

    lane7 = Lane([(1420, 620), (1920, 875), (1647, 880), (1245, 645)])
    lane8 = Lane([(1245, 645), (1647, 880), (1400, 880), (1085, 666)])
    lane9 = Lane([(1085, 666), (1400, 880), (1185, 880), (950, 685)])

    lanes.append(lane1)
    lanes.append(lane2)
    lanes.append(lane3)

    lanes.append(lane4)
    lanes.append(lane5)
    lanes.append(lane6)

    lanes.append(lane7)
    lanes.append(lane8)
    lanes.append(lane9)

    return lanes

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#model = YOLO('ultralytics/yolov5', 'yolov5s', pretrained=True)

#model = YOLO('yolov8n.pt')  # load an official model
#model = YOLO('./weights/best.pt')  # load a custom model
model.conf = 0.16


def getLane9Mask():
    #points = [(1200, 850), (1300, 830), (1340, 860), (1227, 846)]
    #points = [(1180, 850), (1300, 830), (1360, 860), (1250, 880)]
    points = [(1140, 820), (1270, 790), (1400, 880), (1200, 880)]
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    return pts

def getImages():
    imgs = []
    path = './input/'
    for filename in os.listdir(path):
        if os.path.isfile(path + filename) and filename.split(".")[1] == "jpg":
            img = cv2.imread(path + filename)
            imgs.append(img)
    return imgs


def getVehicleNames():
    vehicles = ["car","truck","bycicle","motorcycle","bus"]
    return vehicles

def getVehicles(img):

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = getLane9Mask()

    cv2.fillPoly(mask, [points], color=255)

    mask = cv2.bitwise_not(mask)

    img = cv2.bitwise_and(img, img, mask=mask)

    results = model(img)

    class_labels = model.module.names if hasattr(model, 'module') else model.names

    vehicles = getVehicleNames()

    list = []

    for detection in results.pandas().xyxy[0].iterrows():
        xmin, ymin, xmax, ymax, confidence, class_index = detection[1][:6]
        #half the bounding box down
        ymin = ymin - (ymin-ymax)//2
        class_name = class_labels[int(class_index)]
        if class_name in vehicles:
            list.append(Vehicle(xmin, ymin, xmax, ymax, confidence, class_index))

    return list

def checkLanes(img, show_im = False):

    cars = getVehicles(img)
    lanes = getLanes()

    #score for each lane for each car
    scores = []

    for car in cars:
        #score for each lane
        lane_score = []
        for lane in lanes:
            carBound = Polygon([(car.xmin, car.ymin), (car.xmin, car.ymax), (car.xmax, car.ymax), (car.xmax, car.ymin)])
            points = lane.points
            laneBound = Polygon(points)
            intersection = carBound.intersection(laneBound)
            if not(intersection.is_empty):
                lane_score.append(intersection.area/carBound.area)
            else:
                lane_score.append(0)
        scores.append(lane_score)

    occupied_lanes = [0 for i in range(len(lanes))]
    for lane_scores in scores:
        maximum = max(lane_scores)
        if maximum != 0 and maximum > 0.1:
            largest_index = lane_scores.index(maximum)
            occupied_lanes[largest_index] = 1

    print(occupied_lanes)
    if show_im == True:
        imgs = getImages()

        for car in cars:
            pts = [(car.xmin, car.ymin), (car.xmin, car.ymax), (car.xmax, car.ymax), (car.xmax, car.ymin)]
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (255, 0, 0), 5)

        show_image(img)

    return occupied_lanes

def checkLanesQueries(path = './inputs/', solution_folder = "results/"):

    imgs = []
    queries = []
    filenames = []

    pairs = []

    for filename in os.listdir(path):
        if os.path.isfile(path + filename):
            if filename.split(".")[1] == "jpg":
                img = cv2.imread(path + filename)
                imgs.append(img)
                filenames.append(filename.split(".")[0])
            else:
                f = open(path + filename)
                lines = f.readlines()
                queries.append(lines)


    #imgs = [imgs[0]]

    for i in range(len(imgs)):
        pairs.append((imgs[i],queries[i], filenames[i]))



    for i in range(len(pairs)):
        img = pairs[i][0]
        query = pairs[i][1]
        filename = pairs[i][2]

        query_lanes = []

        no_lanes = query[0].rstrip()
        for i in range(1, int(no_lanes) + 1):
            # print(i)
            # print(query[i].rstrip())
            query_lanes.append(int(query[i].rstrip()))

        occupied = checkLanes(img)

        f = open(solution_folder + filename + "_predicted.txt", "w")
        f.write(no_lanes + "\n")
        for i in range(0, int(no_lanes)):
            # print("i is")
            # print(i)
            # print(query_lanes[i])
            f.write(str(query_lanes[i]) + " " + str(occupied[query_lanes[i] - 1]) + "\n")

def deleteTrailingNewLines(results_folder = "results/"):

    for filename in os.listdir(results_folder):

        # Step 1: Open the file for reading
        with open(results_folder + filename, "r") as file:
            contents = file.read()

        # Step 2: Check for a new line character at the end
        if contents.endswith("\n"):
            # Step 3: Truncate the last character
            with open(results_folder + filename, "w") as file:
                file.write(contents[:-1])


checkLanesQueries()
deleteTrailingNewLines()


































