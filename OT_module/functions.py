from cv2 import line
import numpy as np
from functools import cmp_to_key
import cv2

# from torch._C import int32

def get_another_xPoint(grad, x, y, givenY):
    return (grad * x - y + givenY) / grad
    
def get_line_equation(lane_point):
    # Y = kX + b
    # line_equation: k,b 
    # calculate equation of two lanes.
    line_equation = []
    for i in range(0, len(lane_point), 2):
        m = float(lane_point[i][0]) - float(lane_point[i+1][0])
        k=0
        b=0
        if m == 0:
            k = 10000.0
            b = lane_point[i][1] - k * lane_point[i][0]
        else:
            k = (float(lane_point[i][1]) - float(lane_point[i+1][1])) / m
            b = lane_point[i][1] - k * lane_point[i][0]
        line_equation.append([k,b])
    return line_equation

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

class Lane:
    def __init__(self):
        self.lane_point = [[0,0],[0,0],[0,0],[0,0]]
        self.gradientChangeLeft = 0 
        self.gradientChangeRight = 0
        self.missing_lane_flag = False
        self.boundingBoxes = []
    
    def cmp(self, a, b): # compare func for sorting.
        if(a[1] == b[1]):
            if(a[0] < b[0]):
                return -1
            else:
                return 1
        else:
            if(a[1] > b[1]):
                return 1
            else:
                return -1

    def update_lane_point(self, l, r):
    
        if (len(l) == len(r) and len(l) == 0): # clear vector to zero.
            return 
            # x =[0,0]
            # return {}
            # for i in range(len(self.lane_point)):
            #     self.lane_point[i] = x;
            # return
        
        # sorting according to specific order.
        l.sort(key=cmp_to_key(self.cmp))
        r.sort(key=cmp_to_key(self.cmp))
        left = len(l)
        right = len(r)
        # handle each case.

        # TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO
        if(right == 0):
            # self.lane_point[0] = l[left/4];
            # self.lane_point[1] = l[(left/2 + left)/2];
            self.lane_point[0] = l[0]
            self.lane_point[1] = l[left//2]
        
        elif(left == 0):
            self.lane_point[2] = r[right//4]
            self.lane_point[3] = r[(right//2 + right)//2]
        
        else:
            self.lane_point[0] = l[left//4]
            self.lane_point[1] = l[(left//2 + left)//2]
            self.lane_point[2] = r[right//4]
            self.lane_point[3] = r[(right//2 + right)//2]
        

    def get_lane_point(self):
        return self.lane_point

    def update_boundingBox(self, data):
        self.boundingBoxes = data

    def get_boundingBox(self):
        return self.boundingBoxes
    
    def missing_lane(self):
        self.missing_lane_flag = False
        for i in range(len(self.lane_point)):
            if self.lane_point[i][0] == 0:
                self.missing_lane_flag = True
            if self.missing_lane_flag is True:
                break
        return self.missing_lane_flag

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
        self.lane_available_list = None
        self.lane_threshold_arr = None
        self.ten_pt_threshold = np.zeros(10,dtype=np.int32)
        self.variant = 0.5
        self.detect_result = (False, False)

    def set_center(self, pt):
        self.center = pt

    def set_working_wtate(self, flag):
        self.running = flag

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
    
    def overlap_ratio(self, bb, lane, flag):
        '''
        bb[0]: left x coordinate of bounding box
        bb[1]: right x coordinate of bounding box
        '''
        # print(bb, lane)
        bbWid = bb[1] - bb[0] # calculate bottom width of bounding box.
        ratio = 0
        if(bb[0] < lane[0]): # bounding box on the left of lane.
            flag = -1
            diff = float(bb[1]) - float(lane[0])
            if(diff < 0): # no overlapping
                ratio = 0.0
            else:
                ratio = diff/bbWid * 100.0
        
        elif(bb[1] > lane[1]): # bounding box on the right of lane.
            flag = 1
            diff = float(lane[1]) - float(bb[0])
            if(diff < 0): # no overlapping
                ratio = 0.0
            
            else:
                ratio = diff/bbWid * 100.0
        else: # bounding box in the both lane.
            flag = 0
            ratio = 100.0
        return (flag, ratio)

    def set_lane_available_list(self, lane_upper_y):
        self.lane_available_list = np.zeros(10,dtype=np.uint32)
        start_point_y = lane_upper_y
        end_point_y = 550 # setting by manual according image.
        devide = (end_point_y - start_point_y) / 9
        for i in range(len(self.lane_available_list)):
            if(i==0):
                self.lane_available_list[i] = int(start_point_y)
            else:
                self.lane_available_list[i] = int(self.lane_available_list[i-1] + devide)
    
    def detect_lane_available(self, lane_mask, lane_points):
        if self.lane_available_list is None:
            return (False, False)
        equations = get_line_equation(lane_points)
        pt_l = np.zeros(10,dtype=np.int32)
        pt_r = np.zeros(10,dtype=np.int32)
        # print(equations)
        for i in range(len(self.lane_available_list)):
            accord_left_x = int(round((self.lane_available_list[i] - equations[0][1]) / equations[0][0]))
            accord_right_x = int(round((self.lane_available_list[i] - equations[1][1]) / equations[1][0]))

            self.ten_pt_threshold[i] = max(accord_right_x - accord_left_x ,0)
            bias = 10 # calculate sum of left and right not from center of line

            sum_left = 0
            sum_right = 0
            # calculate mask width left
            for j in range(accord_left_x-bias, 0,-1):                
                if lane_mask[self.lane_available_list[i]][j][2] > 0:
                    sum_left += 1
                else:
                    break
            # calculate mask width left
            for j in range(accord_right_x+bias, lane_mask.shape[1]):
                if lane_mask[self.lane_available_list[i]][j][2] > 0:
                    sum_right += 1
                else:
                    break
            pt_l[i] = sum_left
            pt_r[i] = sum_right
            
            # print(sum_left, sum_right)
        res_l = True
        res_r = True
        for i in range(len(pt_l)):
            if(pt_l[i] < self.ten_pt_threshold[i] * self.variant):
                res_l = False
                break
        for i in range(len(pt_r)):
            if(pt_r[i] < self.ten_pt_threshold[i] * self.variant):
                res_r = False
                break
        self.detect_result = (res_l, res_r)
        
    def detect_overtaking(self, bbs, lane_point):
        line_equation = get_line_equation(lane_point)
        ptarr = [0,0]
        obj_flag = False
        left_flag = False
        right_flag = False
        overlapping_ratio = 40.0
        '''
        threshold_len = 10
        init_threshold = 0.9
        top = lane_point[0][1]
        offset = (lane_point[1][1] - lane_point[0][1]) / 10
        if self.lane_threshold_arr is None:
            self.lane_threshold_arr = [0 for i in range(threshold_len)]
        
            
        diff = 0.0
        for i in range(threshold_len):
            # get according x of each threshold.
            left_laneX = get_another_xPoint(line_equation[0][0], lane_point[0][0], lane_point[0][1], lane_point[1][1] - offset * i)
            right_laneX = get_another_xPoint(line_equation[1][0], lane_point[2][0], lane_point[2][1], lane_point[3][1] - offset * i)

            if(i == 0): # init threshold.
                self.lane_threshold_arr[i] = init_threshold
            else:
                ratio = abs(float(right_laneX) - float(left_laneX)) / diff
                self.lane_threshold_arr[i] = self.lane_threshold_arr[i-1] * ratio

            diff = abs(float(right_laneX) - float(left_laneX))
        '''
        # caluculate each bounding box whether occupy one of side's lane.
        for i in range(0,len(bbs),2):
            # check the bounding box is go through the end.
            if bbs[i][1] == 0 and bbs[i+1][1] == 0:
                break

            for j in range(len(line_equation)): # calculate according x point. y = ax + b => x = (y-b) / a
                ptarr[j] = (bbs[i+1][1] - line_equation[j][1]) / line_equation[j][0]
            
            flag = 0
            # bounding box & two lane width to calculate overlap ratio.
            (flag, ratio) = self.overlap_ratio([bbs[i][0], bbs[i+1][0]], ptarr, flag)
            if(flag == 0 or ratio > overlapping_ratio): # there have car in the same lane.
                obj_flag = True
                break
            '''
            # if(flag != 0): # if in same lane have car, detect both side of car.
            #     box_center = [0,0]
            #     box_center[0] = (bbs[i][0] + bbs[i+1][0]) / 2;
            #     box_center[1] = (bbs[i][1] + bbs[i+1][1]) / 2;
            #     # Y = k * X + b;
            #     # lineEquation[i][0] -> k
            #     # lineEquation[i][1] -> b
                
            #     # calculate the distance between lane and center of bounding box.
            #     # landidx: 0 => left lane, 1 => right lane
            #     lane_idx = (1 if flag == 1 else 0)

            #     length = abs(line_equation[lane_idx][0] * box_center[0] - box_center[1] + line_equation[lane_idx][1]) / np.sqrt(float(box_center[0] * box_center[0] + box_center[1] * box_center[1]))
            #     if box_center[1] < top:
            #         continue
            #     level = (box_center[1] - top) / offset
            #     # print(length, lane_threshold_arr[int(threshold_len-level-1)])
            #     if(length < self.lane_threshold_arr[int(threshold_len-level-1)]): # side car in the threshold.
            #         if(flag == 1):
            #             right_flag = True
            #         elif(flag == -1):
            #             left_flag = True
            '''
        # print(obj_flag, left_flag, right_flag)
        res_flag = 0
        if(obj_flag):
            self.running = True
            '''
            -1: left side can do overtaking.
            0: no one of side can do overtaking
            1: right side can do overtaking.
            '''
            
            # print(self.detect_result)
            (left_flag, right_flag) = self.detect_result
            if(left_flag and right_flag):
                res_flag = 1
            elif(left_flag):
                res_flag = -1
            elif(right_flag):
                res_flag = 1
            else:
                 res_flag = 0
        
        else: # no car in this lane or too far to detect
            self.running = False
            res_flag = 0
        self.set_msg(res_flag)
        
