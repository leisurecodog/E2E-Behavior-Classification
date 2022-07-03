import numpy as np
import os
import cv2
from MOT_module.MOT import MOT
from scipy import stats

def label(mot, vid_name, images):
    strs = '0 means that can not overtake,\n1/2/3 means that overtake from left/dont need/from right.'
    print(strs)
    save_path = './labels'
    save_file_name = vid_name + '.txt'
    final_file_path = os.path.join(save_path, save_file_name)
    total_len = len(images)
    if os.path.exists(final_file_path):
        return 
        
    with open(final_file_path, 'w') as f:
        objdet_dict = {}
        for frame_id, image in enumerate(images):
            if frame_id % 5 == 0:
                ip = os.path.join(dir_p, image)
                raw_image_l = os.path.join(raw_image_p, image)
                frame_depth = cv2.imread(ip)
                frame = cv2.imread(raw_image_l)
                mot.run(frame, objdet_dict)
                # print(objdet_dict)
                det_res = objdet_dict[mot.frame_id-1]
                m_data = 0
                # print(det_res)
                if det_res is not None:
                    for bbox in det_res:
                        b = [int(bbox[i]) for i in range(4)]
                        fcp = frame_depth.copy()
                        sub_sec = fcp[b[1]:b[3], b[0]:b[2]]
                        d = sorted(sub_sec[:,:,0].flatten())
                        val = [v for v in d if v > 0]
                        if len(val) == 0:
                            val = 0
                        else:
                            val = val[0]
                        # idx = np.unravel_index(np.median(sub_sec), sub_sec.shape)
                        # print(val)
                        cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0, 255, 0), 2)
                        # frame = cv2.addWeighted(frame, 0.5, frame_depth, 0.5, 1)
                        cv2.putText(frame, "{}".format(val), (b[0],b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "{}".format(b[2]-b[0]), (b[0],b[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                cv2.imshow(vid_name, frame)
                if cv2.waitKey(0) == 27:
                    return 
                # try:
                #     label = input("({}/{})Label it: ".format(frame_id, total_len))
                #     if label == 'continue':
                #         return final_file_path
                #     labels = label.split(' ')
                #     f.write("{}, {}, {}\n".format(frame_id, labels[0], labels[1]))
                # except KeyboardInterrupt:
                #     os.remove(final_file_path)
                
if __name__ == '__main__':
    depth_folder = '/media/rvl/D/Work/fengan/Dataset/KITTI/depth/data_depth_annotated/train'
    mot = MOT()
    dirs = sorted(os.listdir(depth_folder))
    for dir_name in dirs:
        
        st_idx = [i for i in range(len(dir_name)) if dir_name.startswith("_", i)]
        dir_date = dir_name[:st_idx[2]]
        dir_p = os.path.join(depth_folder, dir_name, 'proj_depth/groundtruth/image_02')
        images = sorted(os.listdir(dir_p))
        # input()
        base_imge_p = '/media/rvl/D/Work/fengan/Dataset/KITTI/raw_data'
        raw_image_p = os.path.join(base_imge_p, dir_name, dir_date, dir_name, 'image_02', 'data')
        # print(raw_image_p)
        # input()
        if not os.path.exists(raw_image_p):
            continue
        res = label(mot, dir_name, images)
        if type(res) is str:
            os.remove(res)
        cv2.destroyAllWindows()
        
