from TP_module.util import prGreen
import time
import numpy as np
import torch
import cv2

class OT:    
    def __init__(self):
        msg = "Initializing OT Module..."
        prGreen(msg)
        from OT_module.opt import set_opt
        from OT_module.yolact_edge_project.eval import load_yolact_edge, prep_display
        from OT_module.yolact_edge_project.utils.augmentations import FastBaseTransform
        from OT_module.PInet.test import PInet_test
        from OT_module.PInet.agent import Agent
        import OT_module.functions as functions

        # Yolact setting
        self.yolact_model = load_yolact_edge()
        self.yolact_model.detect.use_fast_nms = True
        self.transform = FastBaseTransform
        self.get_mask = prep_display
        self.moving_statistics = {"conf_hist": []}
        self.extras = {"backbone": "full", "interrupt": False,\
             "keep_statistics": False,"moving_statistics": self.moving_statistics}
        
        # PInet setting
        self.lane_predict = PInet_test
        self.lane_agent = Agent()
        self.lane_agent.cuda()
        self.lane_agent.evaluate_mode()
        self.lane_agent.load_weights(895, "tensor(0.5546)")
        
        # OTS setting
        self.OTS = functions.overtaking_system()
        self.OT_args = set_opt()
        self.frame_id = 0
        self.counter = 0
        self.exe_time = 0

    def run(self, frame, objdet, test=False):
        st = time.time()
        # frame.shape: [Height, Width, Channels]
        center_x = frame.shape[1] / 2 # get image center
        self.OTS.trg_img_width = frame.shape[1]
        # Get lane line
        # x_coord, y_coord = [], []
        # 偵測車道線
        x_coord, y_coord = self.lane_predict(self.lane_agent, frame)
        t1 = time.time()
        # 更新車道線
        if len(x_coord) > 0 and len(y_coord) > 0:    
            self.OTS.set_lane([x_coord, y_coord], center_x)
        t2 = time.time()

        # execute ot detect when both lane is detected.
        if self.OTS.both_lane_flag:
            
            # Yolact pre-stage
            t3 = time.time()
            frame_tensor = torch.from_numpy(frame).cuda().float()
            batch = self.transform()(frame_tensor.unsqueeze(0))
            with torch.no_grad():
                net_outs = self.yolact_model(batch, extras=self.extras) # yolact edge detect lane mask
            preds = net_outs["pred_outs"]
            # Get lane mask
            lane_mask = self.get_mask(preds, frame_tensor, None, None, undo_transform=False, class_color=True)
            
            # 偵測超車
            self.OTS.detect_overtaking(objdet, lane_mask, frame)
            # test flag為True時會把車道線畫出來做debug
            if test:
                for i in range(len(self.OTS.left_lane[0])):
                    cv2.circle(frame, (int(self.OTS.left_lane[0][i]), int(self.OTS.left_lane[1][i])), 3, (0, 0, 255), -1)
                for i in range(len(self.OTS.right_lane[0])):
                    cv2.circle(frame, (int(self.OTS.right_lane[0][i]), int(self.OTS.right_lane[1][i])), 3, (0, 0, 255), -1)
                # frame = cv2.addWeighted(lane_mask, 1, frame, 1, 0.0)
            self.counter += 1
            self.exe_time += (time.time() - st)
            t4 = time.time()
            # print(t4-t3, t2-t1, t1-st)
        else:
            self.OTS.set_msg(0)
        
        self.frame_id += 1
        if test:
            return frame
        