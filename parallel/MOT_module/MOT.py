from TP_module.util import prGreen
from system_util import ID_check
import time

class MOT:
    def __init__(self):
        msg = "Initializing MOT Module..."
        prGreen(msg)
        # import necessary package
        from MOT_module.tools.demo_track_yolov5 import make_parser as parser_MOT
        from MOT_module.tracker.byte_tracker import BYTETracker
        from MOT_module.tools.demo_track_yolov5 import load_yolov5

        self.MOT_args = parser_MOT().parse_args()
        self.tracker = BYTETracker(self.MOT_args, frame_rate=self.MOT_args.fps)
        self.object_predictor, self.imgsz, self.names = load_yolov5(rt=True)
        # self.counter = 0
        # self.exe_time = 0
        # self.yolo_time = 0
        # self.tracker_time = 0
        # self.yolo_counter = 0
        # self.tracker_counter = 0
        
        self.frame_id = 0
        
        # self.format = 'bbox'


    def run(self, frame, dict_objdet, lock):
        from MOT_module import yolo_detect
        st = time.time()
        img_info = {}
        self.current_MOT = None
        t1 = time.time()
        outputs = yolo_detect.detect(self.object_predictor, self.imgsz, self.names, frame)
        self.yolo_time += (time.time()-t1)
        self.yolo_counter += 1
        res = outputs.cpu().detach().numpy() if outputs is not None else None
        dict_objdet.update({self.frame_id:res})
        img_info['height'], img_info['width'] = frame.shape[:2]
        img_info['raw_img'] = frame

        if outputs is not None:
            self.current_MOT = {}
            t1 = time.time()
            online_targets = self.tracker.update(reversed(outputs), [img_info['height'], img_info['width']], [img_info['height'], 
            img_info['width']])
            self.tracker_time += (time.time()-t1)
            self.tracker_counter += 1
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                self.current_MOT[tid] = tlwh[:4]
        else:
            print("MOT outputs is None.")
        self.frame_id += 1
        # self.counter += 1
        # self.exe_time += (time.time() - st)
        