# from cProfile import label
import numpy as np
import os
import cv2
from MOT_module.MOT import MOT
from OT_module.main import inference, OTS

best_acc = 0
def update_result(acc, overlap_r, var):
    global best_acc
    if acc > best_acc:
        best_acc = acc
        update_result.olap_r = overlap_r
        update_result.var = var
        print("Update Best Accuracy: {} ,using these param: olap {} & var {}.".format(acc, overlap_r, var))


dict_msg = {"You can't overtake.":0, "You can overtake from left side.": 1,\
     "You don't need to overtake.":2, "You can overtake from right side.":3}

def test_OT(label_path, mot):
    video_path = label_path.replace('label', 'front')
    label_files = sorted(os.listdir(label_path))
    for file in label_files:
        dict_label = {}
        file_path = os.path.join(label_path, file)
        tmp = []
        with open(file_path, 'r') as f:
            tmp = f.readlines()
        for line in tmp:
            l_list = line.split(',')
            # print(l_list)
            # 0 means cannot overtake, 1/2/3 mean overtake from left/don't need/right
            label = l_list[1].replace('\n','')
            dict_label[int(l_list[0])] = int(label) 
        cap = cv2.VideoCapture(os.path.join(video_path, file.replace('txt', 'MOV')))
        dict_frame = {}
        fid = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if fid != 0 and fid % 10 == 0:
                    frame = cv2.resize(frame, (1080, 720))
                    dict_frame[fid] = frame
                fid += 1
            else:
                break
        o_rate_iter = np.arange(0, 100, 10)
        var_iter = np.arange(0, 1, 0.1)
        for r1 in o_rate_iter:
            for r2 in var_iter:
                OTS.overlap_rate = r1
                OTS.variant = r2
                # print("OTS.overlap_rate, OTS.variant:  ", OTS.overlap_rate, OTS.variant)
                acc_time = 0
                frame_id = 10
                while frame_id in dict_frame:
                    fm = dict_frame[frame_id]
                    mot.run(fm)
                    msg = inference(mot.objdet, fm)
                    # print(dict_msg[msg], dict_label[frame_id])
                    acc_time += 1 if dict_msg[msg] == dict_label[frame_id] else 0
                    frame_id += 10
                acc = acc_time / len(dict_label)
                update_result(acc, OTS.overlap_rate, OTS.variant)

if __name__ == '__main__':
    mot = MOT()
    # ot = OT()
    path = '/media/rvl/D/Work/fengan/Dataset/CEO/20201116/label'
    test_OT(path, mot)
