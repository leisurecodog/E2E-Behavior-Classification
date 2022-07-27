import numpy as np
import cv2
from BC_module import BC
import os

def tester():
    module_BC = BC()
    path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/bdd100k_ano_half/images/track/train'
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        files = sorted(os.listdir(folder_path))
        for file in files:
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            h, w = img.shape[:2]
            label_path = file_path.replace('images', 'labels_with_ids').replace('jpg', 'txt')
            tmp = []
            with open(label_path, 'r') as f:
                tmp = f.readlines()
            dics = {}
            for line in tmp:
                lst = line.split(' ')
                lst = [float(d) for d in lst]
                
                ID = int(lst[1])
                cen_x, cen_y = lst[2] * w, lst[3] * h
                off_x, off_y = lst[4] * w, lst[5] * h
                dics[ID] = [cen_x-off_x/2, cen_y-off_y/2, off_x, off_y]

if __name__ == '__main__':
    tester()