# from cProfile import label
import numpy as np
import os
import cv2
from OT_module.OT import OT
from sklearn.metrics import classification_report, confusion_matrix

dict_msg = {"You can't overtake.":0, "You can overtake from left side.": 1,\
     "You don't need to overtake.":2, "You can overtake from right side.":3}
key_list = list(dict_msg.keys())
val_list = list(dict_msg.values())
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
            # print(tmp)
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

    o_rate_iter = np.arange(100, 0, -10)
    var_iter = np.arange(1, 0, -0.1)
    debug_flag = False
    gt_total_label = {0:0, 1:0, 2:0, 3:0}
    pred_total_label = {0:0, 1:0, 2:0, 3:0}
    for r1 in o_rate_iter:
        for r2 in var_iter:
            module_OT.OTS.overlap_rate = 10
            module_OT.OTS.variant = 0.8
            avg_acc = 0
            avg_acc_ano = 0
            avg_acc_have_lane = 0
            avg_acc_nowidth = 0
            total_pred = []
            total_gt = []
            total_hl_pred = []
            total_hl_gt = []
            target_names = ['cannot overtake', 'overtake from left',\
                 'don\'t need to overtake', 'overtake from right']
            for k, dict_label in dict_labels.items():
                pred = []
                pred_nowidth = []
                pred_same = []
                pred_have_laneline = []
                hl_pred = []
                hl_gt = []
                gt = []
                dict_frame = dict_videos[k]
                frame_id = 5
                while frame_id in dict_label:
                    fm = dict_frame[frame_id]
                    outputs = yolo_detect.detect(object_predictor, imgsz, names, fm)
                    if outputs is None:
                        frame_id += offset
                        continue
                    outputs = outputs.cpu().detach().numpy()
                    fm = module_OT.run(fm, outputs, test=debug_flag)
                    msg = module_OT.OTS.msg
                    if debug_flag:
                        for b in outputs:
                            nb = [int(d) for d in b[:4]]
                            cv2.rectangle(fm, (nb[0], nb[1]), (int(nb[2]), int(nb[3])), (255,255,255), 2)
                            # cv2.putText(fm, str(), (nb[0], nb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        gt_msg = key_list[val_list.index(dict_label[frame_id])]
                        
                        cv2.putText(fm, gt_msg, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(fm, msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('t', fm)
                        cv2.imwrite('./tt.png', fm)
                        cv2.waitKey(0)

                    # print(dict_msg[msg], dict_label[frame_id])
                    # acc_time += 1 if dict_msg[msg] == dict_label[frame_id] else 0
                    pred.append(dict_msg[msg])
                    pred_nowidth.append(module_OT.OTS.res_flag_nowidth)
                    gt_total_label[int(dict_label[frame_id])] += 1
                    gt.append(dict_label[frame_id])
                    # make can't overtake as same as don't need to overtake.
                    pred_label = dict_msg[msg]
                    gt_label = dict_label[frame_id]
                    if module_OT.OTS.both_lane_flag:
                        pred_have_laneline.append(pred_label == gt_label)
                        hl_pred.append(pred_label)
                        hl_gt.append(gt_label)
                    if (pred_label == 0 or pred_label == 2) and (gt_label == 0 or gt_label == 2):
                        pred_same.append(gt_label)
                    else:
                        pred_same.append(pred_label)

                    frame_id += offset
                correct0 = np.array(pred_nowidth) == np.array(gt)
                correct1 = np.array(pred) == np.array(gt)
                correct2 = np.array(pred_same) == np.array(gt)
                avg_acc_nowidth += correct0.sum() / len(correct0)
                avg_acc += correct1.sum() / len(correct1)
                avg_acc_ano += correct2.sum() / len(correct2)
                
                avg_acc_have_lane += np.array(pred_have_laneline).sum() / len(pred_have_laneline)
                total_hl_pred.extend(hl_pred)
                total_hl_gt.extend(hl_gt)
                total_pred.extend(pred)
                total_gt.extend(gt)
            # print(total_gt)
            print(confusion_matrix(total_gt, total_pred))
            print(confusion_matrix(total_hl_gt, total_hl_pred))

            avg_acc /= len(dict_labels)
            avg_acc_ano /= len(dict_labels)
            avg_acc_nowidth /= len(dict_labels)
            avg_acc_have_lane /= len(dict_labels)
            disp_str = "Overlap_ratio: {} \t var: {} \t acc: {} \t nolane acc: {} \t have lane: {}"\
                .format(round(r1, 4), round(r2, 4), round(avg_acc, 4),\
                     round(avg_acc_nowidth, 4), round(avg_acc_have_lane, 4))
            print(disp_str)
            
        # print(gt_total_label)
        # return 
            

if __name__ == '__main__':
    
    path = '../../OT_label_Depth/labels'
    # path = 'labels_backup'
    test_OT(path)
