from TP_module.util import prGreen
import time
import numpy as np
import torch

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
        
        self.OTS = functions.overtaking_system()
        self.OT_args = set_opt()
        self.counter = 0
        self.exe_time = 0

    def run(self, frame, frame_id, dict_objdet):
        # st = time.time()
        if frame_id in dict_objdet:
            objdet = dict_objdet[frame_id]
            # frame.shape: [Height, Width, Channels]
            center_x = frame.shape[1] / 2 # get image center

            # Get lane line
            x_coord, y_coord = self.lane_predict(self.lane_agent, frame)
            if len(x_coord) > 0 and len(y_coord) > 0:
                self.OTS.set_lane([x_coord, y_coord], center_x)

            # Yolact pre-stage
            frame_tensor = torch.from_numpy(frame).cuda().float()
            batch = self.transform()(frame_tensor.unsqueeze(0))
            with torch.no_grad():
                net_outs = self.yolact_model(batch, extras=self.extras) # yolact edge detect lane mask
            preds = net_outs["pred_outs"]
            # Get lane mask
            lane_mask = self.get_mask(preds, frame_tensor, None, None, undo_transform=False, class_color=True)

            if self.OTS.both_lane_flag:
                # execute ot detect when both lane is detected.
                t1 = time.time()
                self.OTS.detect_overtaking(objdet, lane_mask, frame)
                t2 = time.time()
                print("detect time:", t2-t1)
                self.counter += 1
                # self.exe_time += (time.time() - st)
            else:
                self.OTS.msg = "You can't overtake."

    def display_lane(self):
        print("Display Lane")
        # draw lane after process
        # if OTS.both_lane_flag and draw_laneline_flag:
            # for i in range(len(OTS.left_lane[0])):
                # x, y = OTS.left_lane[:, i]
                # cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
            # for i in range(len(OTS.right_lane[0])):
                # x, y = OTS.right_lane[:, i]
                # cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
        # for i in range(len(objdet)):
            # x1, y1 = objdet[i][:2]
            # x2, y2 = objdet[i][2:4]
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255))