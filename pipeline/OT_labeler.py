import numpy as numpy
import os
import cv2
from MOT_module.MOT import MOT
def label(mot, path, save_path):

    # Label overtake data each 10 frame
    interval = 10
    if os.path.exists(save_path):
        return 

    cap = cv2.VideoCapture(path)
    frame_id = 0 # important
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    
    try:
        f = open(save_path, 'w')
        while True:    
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame,(1080, 720))
                
                # print(frame_id)
                if frame_id != 0 and frame_id % interval == 0:
                    mot.run(frame)
                    det_res = mot.objdet
                    for det in det_res:
                        bbox = det[:4]
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                    cv2.imshow('t', frame)
                    cv2.waitKey(1)
                    res = input("0 \t means cannot overtake,\n1/2/3 \t means that overtake from left/center/right:\n")
                    f.write("{},{}\n".format(frame_id, res))
                frame_id += 1
            else:
                break
        f.close()
    except KeyboardInterrupt:
        import sys
        if os.path.exists(save_path):
            os.remove(save_path)
        sys.exit()


if __name__ == '__main__':
    '''
    (using yolov5s to detect object.)
    (detected by yolov5s.)
     (by yolov5s and) eye 

    label criteria:
        ego lane occlusion: 
            Have object in Ego lane: occlusion.
            No object in Ego lane: don't need to overtake.
        can't overtake: 
            both side have object 
        one of side lane can be passed: 
            no object in one of lane.

    '''
    
    folder = '/media/rvl/D/Work/fengan/Dataset/CEO/20201116/front'
    label_folder = folder.replace('front', 'label')
    mot = MOT()
    vids = os.listdir(folder)
    for name in vids:
        path = os.path.join(folder, name)
        save_path = os.path.join(label_folder, name.replace('MOV', 'txt'))
        label(mot, path, save_path)
        
