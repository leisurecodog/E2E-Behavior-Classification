from TP_module.util import prGreen
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
        self.counter = 0
        self.exe_time = 0

    def run(self, frame, frame_id, format):
        from MOT_module import yolo_detect
        st = time.time()
        img_info = {}
        results = {}
        outputs = yolo_detect.detect(self.object_predictor, self.imgsz, self.names, frame)
        self.objdet = outputs.cpu().detach().numpy()
        img_info['height'], img_info['width'] = frame.shape[:2]
        img_info['raw_img'] = frame

        if outputs is not None:
            online_targets = self.tracker.update(reversed(outputs), [img_info['height'], img_info['width']], [img_info['height'], 
            img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

                if format == 'bbox':
                    results[tid] = tlwh[:4]
                else:
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f}\
                        ,{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
        else:
            print("MOT outputs is None.")
        self.result = results
        # print("MOT time: ", time.time()-st)
        self.counter += 1
        self.exe_time += (time.time() - st)
        