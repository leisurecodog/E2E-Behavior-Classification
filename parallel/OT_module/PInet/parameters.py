#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np

class Parameters():
    model_path = "OT_module/PInet/weights/"
    x_size = 512
    y_size = 256
    resize_ratio = 8
    grid_x = x_size/resize_ratio  #64 
    grid_y = y_size/resize_ratio  #32 
    feature_size = 4
    mode = 1
    threshold_point = 0.6 # or 0.81
    threshold_instance = 0.22

    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((int(grid_y), int(grid_x), 2))
    for y in range(int(grid_y)):
        for x in range(int(grid_x)):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
