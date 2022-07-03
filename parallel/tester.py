# from cProfile import label
import numpy as np
import os
import cv2
from OT_module.OT import OT

dict_msg = {"You can't overtake.":0, "You can overtake from left side.": 1,\
     "You don't need to overtake.":2, "You can overtake from right side.":3}

def test_OT(label_path):
    from MOT_module.tools.demo_track_yolov5 import load_yolov5
    from MOT_module import yolo_detect
    object_predictor, imgsz, names = load_yolov5(rt=True)
    module_OT = OT()
    offset = 5
    # video_path = label_path.replace('label', 'front')
    video_path = '/media/rvl/D/Work/fengan/Dataset/KITTI/raw_data'
    label_files = sorted(os.listdir(label_path))
    dict_videos = {}
    dict_labels = {}
    for file in label_files:
        dict_label = {}
        file_path = os.path.join(label_path, file)
        tmp = []
        # get label and put in dict
        with open(file_path, 'r') as f:
            tmp = f.readlines()
        for line in tmp:
            l_list = line.split(',')
            label = l_list[1].replace('\n','')
            dict_label[int(l_list[0])] = int(label) 
        dict_labels[file.replace('.txt', '')] = dict_label
        print(file)
        # get frame and put in dict
        dict_frame = {}
        vid_folder_name = file.replace('.txt', '')
        vid_name_only_date = vid_folder_name[:10]
        path = os.path.join(video_path, vid_folder_name, vid_name_only_date, vid_folder_name)
        path = os.path.join(path, 'image_02/data')
        image_names = sorted(os.listdir(path))
        for name in image_names:
            nums = int(name.split('.')[0])
            if nums % offset == 0:
                p = os.path.join(path, name)
                img = cv2.imread(p)
                dict_frame[nums] = img
        dict_videos[file.replace('.txt', '')] = dict_frame

    o_rate_iter = np.arange(0, 100, 10)
    var_iter = np.arange(0, 1, 0.1)
    
    for r1 in o_rate_iter:
        for r2 in var_iter:
            module_OT.OTS.overlap_rate = r1
            module_OT.OTS.variant = r2
            # print("OTS.overlap_rate, OTS.variant:  ", OTS.overlap_rate, OTS.variant)
            avg_acc = 0
            avg_acc_ano = 0
            for k, dict_label in dict_labels.items():
                # print(k)
                dict_frame = dict_videos[k]
                acc_time = 0
                acc_time_another = 0
                frame_id = 5
                while frame_id in dict_label:
                    fm = dict_frame[frame_id]
                    outputs = yolo_detect.detect(object_predictor, imgsz, names, fm)
                    if outputs is None:
                        frame_id += offset
                        continue
                    outputs = outputs.cpu().detach().numpy()
                    fm = module_OT.run(fm, outputs, test=False)
                    msg = module_OT.OTS.msg
                    # for b in outputs:
                    #     nb = [int(d) for d in b[:4]]
                    #     cv2.rectangle(fm, (nb[0], nb[1]), (int(nb[2]), int(nb[3])), (255,255,255), 2)
                    #     cv2.putText(fm, str(nb[2]-nb[0]), (nb[0], nb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # cv2.putText(fm, str(dict_label[frame_id]), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # cv2.putText(fm, msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # cv2.imshow('t', fm)
                    # cv2.waitKey(0)
                    # print(dict_msg[msg], dict_label[frame_id])
                    acc_time += 1 if dict_msg[msg] == dict_label[frame_id] else 0
                    # make can't overtake as same as don't need to overtake.
                    pred_label = dict_msg[msg]
                    gt_label = dict_label[frame_id]
                    if pred_label == 2:
                        pred_label = 0
                    if gt_label == 2:
                        gt_label = 0
                    acc_time_another += 1 if pred_label == gt_label else 0
                    frame_id += offset
                acc = acc_time / len(dict_label)
                avg_acc += acc
                ano_acc = acc_time_another / len(dict_label)
                avg_acc_ano += ano_acc
            avg_acc = avg_acc / len(dict_labels)
            avg_acc_ano = avg_acc_ano / len(dict_labels)
            disp_str = "Overlap_ratio: {} \t var: {} \t acc: {} \t make_same_acc: {}"\
                .format(round(r1, 4), round(r2, 4), round(avg_acc, 4), round(avg_acc_ano, 4))
            print(disp_str)
            

if __name__ == '__main__':
    
    # path = '../../OT_label_Depth/labels'
    path = 'labels_backup'
    test_OT(path)
