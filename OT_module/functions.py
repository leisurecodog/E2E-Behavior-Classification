from cv2 import line
import numpy as np
from functools import cmp_to_key
import cv2

# from torch._C import int32

class Arrow:
    def __init__(self, wid=None, hei=None):
        self.center_x = 0
        self.center_y = 0
        self.horizonMove = 0
        self.rotateAngle = 0
        self.currentAngle = 0
        self.height = 0
        self.width = 0

    def get_shape(self):
        res = []
        if(self.horizonMove > 0):
            self.center_x += 1
            self.horizonMove -= 1
        
        elif(self.horizonMove < 0):
            self.center_x -= 1
            self.horizonMove =+ 1
        res = [
            [0+self.center_x, 0+self.center_y],
            [-self.width/2 + self.center_x, self.height+self.center_y],
            [0+self.center_x, 0+self.center_y + self.height/2],
            [self.width/2 + self.center_x, self.height+self.center_y]]
        return np.array(res)

    def set_WH(self, wid, hei):
        self.width = wid
        self.height = hei

    def set_position(self, pos):
        if(self.center_x == 0 and self.center_y == 0):
            self.center_x = pos[0]
            self.center_y = pos[1]
        else:
            bias = pos[0] - self.center_x;
            if((self.horizonMove < 0 and bias <0) or (self.horizonMove > 0 and bias > 0)):
                self.horizonMove = (max(self.horizonMove,  bias) if bias < 0 else min(self.horizonMove, bias))
            
            else:
                self.horizonMove += bias
            
    def get_center(self):
        return (self.center_x, self.center_y)

    def get_WH(self):
        return (self.width, self.height)

    def get_rotate_angle(self):
        return self.currentAngle
    
    def update_rotate_angle(self):
        if(self.currentAngle < self.rotateAngle):
            self.currentAngle += 1
        elif(self.currentAngle > self.rotateAngle):
            self.currentAngle -= 1
            
    def set_rotate_angle(self, angle):
        self.rotateAngle = angle

    def stop_move(self):
        self.horizonMove = 0

    def get_color(self, state):
        if(state == False):
            return (255, 255, 0)
        else:
            return (0, 0, 255)
    
    def overtaking_arrow(self, shape):
        w, h = shape
        overtaking_mat = np.zeros((w,h),)

# =========================================================================================

class overtaking_system:
    def __init__(self, x=None, y=None):
        if x is None and y is None:
            self.center = [0,0]
        else:
            self.center = [x,y]
        self.overtake_path = []
        self.running = False
        self.msg = ""
        self.variant = 0.5
        self.detect_result = (False, False)

    def set_center(self, pt):
        self.center = pt

    def set_working_wtate(self, flag):
        self.running = flag

    def update_lane(self):
        self.left_lane = np.array(self.left_lane, dtype=np.int32)
        self.right_lane = np.array(self.right_lane, dtype=np.int32)
        final_top_idx = -1
        for i in range(len(self.left_lane[0])):
            left_x = self.left_lane[0][i]
            right_x = self.right_lane[0][i]

            if left_x > right_x:
                continue
            else:
                final_top_idx = i
                break
        # this parameter need smarter setting
        # TO DO
        screen_bottom = 590
        l_idx , r_idx = self.find_nearest_idx(screen_bottom)

        self.left_lane = self.left_lane[:, final_top_idx:l_idx]
        self.right_lane = self.right_lane[:, final_top_idx:r_idx]

    def set_lane(self, lanes, center):
        self.left_lane = None
        self.right_lane = None
        self.both_lane_flag = False
        l_dist = np.inf
        r_dist = np.inf
        x_coord, y_coord = lanes[0], lanes[1]
        for i in range(len(x_coord)): 
            if x_coord[i][-1] < center:
                if abs(x_coord[i][-1] - center) < l_dist:
                    l_dist = abs(x_coord[i][-1] - center)
                    self.left_lane = [x_coord[i], y_coord[i]]
            elif x_coord[i][-1] > center: 
                if abs(x_coord[i][-1] - center) < r_dist:
                    r_dist = abs(x_coord[i][-1] - center)
                    self.right_lane = [x_coord[i], y_coord[i]]
            else:
                print("Lane {i} Setting Error ".format(i))

        if self.left_lane is not None and self.right_lane is not None:
            self.update_lane()
            self.both_lane_flag = True
            

    def set_msg(self, res_flag):
        if self.running:
            if(res_flag == 0):
                self.msg = "You can't overtake."
            elif(res_flag == 1):
                self.msg = "You can overtake from right side."
            elif(res_flag == -1):
                self.msg = "You can overtake from left side."
        else:
            self.msg = "You don't need to overtake."

    def find_nearest_idx(self, value):
        l_idx = np.abs(self.left_lane[1] - value).argmin()
        r_idx = np.abs(self.right_lane[1] - value).argmin()
        return (l_idx, r_idx)

    def overlap_ratio(self, bb):
        # bb: [left_top, right_bottom, ?, ?]

        flag = 0 # flag: -1 mean bb on left of lane, 0 mean middle, 1 mean right of lane
        bbWid = bb[2] - bb[0] # calculate bottom width of bounding box.
        bb_bottom = bb[3]
        (l_idx, r_idx) = self.find_nearest_idx(bb_bottom)
        ratio = 0
        left_x = self.left_lane[0,l_idx]
        right_x = self.right_lane[0, r_idx]
        
        if(bb[0] < left_x): # bounding box on the left of lane.
            flag = -1
            diff = float(bb[2]) - float(left_x)
            if(diff < 0): # no overlapping
                ratio = 0.0
            else:
                ratio = diff/bbWid * 100.0
        
        elif(bb[2] > right_x): # bounding box on the right of lane.
            flag = 1
            diff = float(right_x) - float(bb[0])
            if(diff < 0): # no overlapping
                ratio = 0.0
            
            else:
                ratio = diff/bbWid * 100.0
        else: # bounding box in the both lane.
            flag = 0
            ratio = 100.0
        return (flag, ratio)

    def detect_lane_available(self):
        split = 10
        devide = len(self.left_lane) // split
        left_lane_idx = [len(self.left_lane) - i*devide for i in range(split)]
        devide = len(self.right_lane) // split
        right_lane_idx = [len(self.right_lane) - i*devide for i in range(split)]
        overlap_left = []
        overlap_right = []
        threshold_list = []
        for i in range(split):
            coord_left_x, coord_left_y = self.left_lane[:, left_lane_idx[i]]
            coord_right_x, coord_right_y = self.right_lane[:, right_lane_idx[i]]
            threshold = coord_right_x - coord_left_x
            threshold_list.append(threshold)
            bias = 5 # calculate sum of left and right not from center of line
            sum_left = 0
            sum_right = 0
            for j in range(coord_left_x-bias, 0, -1):
                if self.lane_mask[coord_left_y][j][2] > 0:
                    sum_left += 1
                else:
                    break
            overlap_left.append(sum_left)
            for j in range(coord_right_x+bias, self.lane_mask.shape[1]):
                if self.lane_mask[coord_right_y][j][2] > 0:
                    sum_right += 1
                else:
                    break
            overlap_right.append(sum_right)
        res_l = True
        res_r = True
        for i in range(split):
            if overlap_left[i] < threshold_list[i] * self.variant:
                res_l = False
                break
        for i in range(split):
            if overlap_right[i] < threshold_list[i] * self.variant:
                res_r = False
                break

        self.detect_result = (res_l, res_r)   

    def detect_overtaking(self, bbs, lane_mask):
        if not self.both_lane_flag:
            # print("Overtaking System is not ready.......")
            self.running = True
            self.set_msg(0)
            return
        self.lane_mask = lane_mask
        overlap_ratio = 40.0
        obj_flag = False
        for i in range(len(bbs)):
            (flag, ratio) = self.overlap_ratio(bbs[i])
            if ratio > overlap_ratio:
                obj_flag = True
                break

        self.detect_lane_available()
        if obj_flag:
            self.running = True
            (l_flag, r_flag) = self.detect_result
            if l_flag and r_flag:
                res_flag = -1
            elif l_flag:
                res_flag = -1
            elif r_flag:
                res_flag = 1
            else:
                res_flag = 0
        else:
            self.running = False
            res_flag = 0
        self.set_msg(res_flag)
