from TP_module.TP import TP
import os
import cv2

def TP_tester():
    module_TP = TP()
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
            # print(dics)
            module_TP.update_traj(dics)
            if module_TP.is_some_id_predictable():
                module_TP.run()
                for ID, traj in module_TP.result.items():
                    if ID in dics:
                        for v in traj:
                            cv2.circle(img, (int(v[0]), int(v[1])), 3, (255, 0, 0), -1)
            for k, v in dics.items():
                x1, y1 = int(v[0]), int(v[1])
                x2, y2 = int(v[0] + v[2]), int(v[1] + v[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.imshow('t', img)
            if cv2.waitKey(0) == 27:
                return 

                


if __name__ == '__main__':
    TP_tester()