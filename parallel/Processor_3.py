import time
import cv2

def run(dict_frame, dict_MOT, dict_traj_current, dict_traj_future, dict_BC, dict_OT, lock):    
    frame_id = 0
    last_frame_time = 0
    while True:
        FPS = 0
        # output: show video if all data is in share dictionary.
        # if frame_id in dict_MOT and frame_id in dict_traj_current and frame_id in dict_traj_future:
        if frame_id in dict_MOT and frame_id in dict_BC:
            entry_time = time.time()
            if frame_id != 0:
                waiting_time = entry_time - last_frame_time
                FPS = 1 / waiting_time
                print("FPS: {}".format(FPS))
            bbox = dict_MOT[frame_id]
            # lock.release()
            fm = dict_frame[frame_id]
            for id in bbox.keys():
                draw_color_idx = 0
                x1, y1, offset_x, offset_y = bbox[id]
                x2, y2 = x1 + offset_x, y1 + offset_y
                # if BC_result_flag and id in self.BC.result.keys():
                #     bcr = self.BC.result[id]
                #     if bcr == -1:
                #         draw_color_idx = 1
                # cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
                cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
                cv2.putText(fm, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                        1, (255, 255, 0), thickness=1)
            for k, traj in dict_traj_current.items():
                for v in traj:
                    cv2.circle(fm, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
            cv2.putText(fm, "FPS: {}".format(FPS), (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
            cv2.imshow('t', fm)
            wk = 1
            if cv2.waitKey(wk) == 27:
                break
            frame_id += 1
            last_frame_time = entry_time