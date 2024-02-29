class Lane:
    def __init__(self, points):
        self.points = points

class Vehicle:
    def __init__(self, xmin, ymin, xmax, ymax, confidence, class_index):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence
        self.class_index = class_index