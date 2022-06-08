import os
import logging
import numpy as np
import cv2
from TP_module.util import prGreen
import torch

class DrivingBehaviorSystem:
    def __init__(self):
        """
        self.traj: list = [frame_1, frame_2, ... , frame_i]
        frame_i: dict = {key: id, value: [x,y] coordinate}
        
        MOT: Multiple Object Tracking
        TP: Trajectory Prediction
        BC: Behaivior Classification
        OT: OverTaking Assistant
        """
        # root_path = os.getcwd()
        self.traj = []
        self.future_trajs = {}
        self.traj_reset_flag = False
        self.init_MOT()
        self.init_TP()
        self.init_BC()
        self.init_OT()
    # ========================= MOT module code =========================
    def init_MOT(self):
        msg = "Initializing MOT Module..."
        prGreen(msg)
        # import necessary package
        from MOT_module.tools.demo_track_yolov5 import make_parser as parser_MOT
        from MOT_module.tracker.byte_tracker import BYTETracker
        from MOT_module.tools.demo_track_yolov5 import load_yolov5

        self.MOT_args = parser_MOT().parse_args()
        self.tracker = BYTETracker(self.MOT_args, frame_rate=self.MOT_args.fps)
        self.object_predictor, self.imgsz, self.names = load_yolov5(rt=True)

    def MOT_run(self, frame, frame_id, format):
        from MOT_module import yolo_detect
        img_info = {}
        results = {}
        outputs = yolo_detect.detect(self.object_predictor, self.imgsz, self.names, frame)
        self.objdet_outputs = outputs.cpu().detach().numpy()
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
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
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
        self.MOT_result = results
    # ========================= TP module code ==========================
    def init_TP(self):
        msg = "Initializing TP Module..."
        prGreen(msg)
        from TP_module.ddpg import DDPG
        from TP_module import env as Env
        from TP_module import parser as TP_parser
        TP_args = TP_parser.get_parser()
        TP_args.actor = "seq2seq"
        print(TP_args.actor)
        model_path = './TP_module/weights/' + TP_args.actor
        # TP_args.actor = 'LSTM'
        self.env = Env.environment(data_select=TP_args.mode, mode='test', args=TP_args)
        self.traj_len_required = self.env.nb_t 
        nb_states = self.env.nb_s
        nb_actions = self.env.nb_a
        self.agent = DDPG(nb_states, nb_actions, TP_args)
        self.agent.load_weights(model_path)
        self.agent.is_training = False
        self.agent.eval()
        self.policy = lambda x: self.agent.select_action(x, decay_epsilon=False)
            
    def update_traj(self):
        data = self.MOT_result
        frame = {}
        for id in data.keys():
            # record ID appear times.

            center_x = data[id][0] + data[id][2] // 2
            center_y = data[id][1] + data[id][3] // 2
            frame[id] = [center_x, center_y]
            if id not in self.ID_counter:
                self.ID_counter[id] = 0
            self.ID_counter[id] += 1
        self.traj.append(frame)
        # print("Update Traj Finished.")

    def predict_traj(self, traj):
        # read last k traj to predict
        init_state = traj[-self.traj_len_required:]
        future_traj = []
        # current_state: [x0, y0, x1, y1, .... x4, y4]
        current_state = np.array(init_state).reshape(-1)
        # get future trajectory that length is the same with trajectory history.
        for _ in range(self.traj_len_required):
            # print(current_state)
            action = self.policy(current_state)
            next_traj = [current_state[-2] + action[0], current_state[-1] + action[1]]
            future_traj.append(next_traj)
            next_state = np.concatenate((current_state[2:], np.asarray(next_traj, dtype=np.float32)), axis=0)
            current_state = next_state
        return future_traj

    def get_future_traj(self):
        if not self.is_enough_tarj():
            return None
        self.future_trajs = {}
        traj_id = []
        for k, v in self.ID_counter.items():
            if v >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                for frame in self.traj:
                    if k in frame:
                        traj_id.append(frame[k])
                # if id not in future_trajs:
                    # future_trajs[id] = []
                # print(traj_id)
                self.future_trajs[k] = self.predict_traj(traj_id)
    
    def is_enough_tarj(self):
        # one of traj has long enough to predict trjectory.
        return True in ( np.array(list(self.ID_counter.values())) > self.traj_len_required)

    def traj_reset(self):
        self.traj = []
        self.ID_counter = {}
        self.future_trajs = {}

    # ========================= BC module code ==========================
    def init_BC(self):
        msg = "Initializing BC Module..."
        prGreen(msg)
        from BC_module.parser import add_parser
        from sklearn.svm import OneClassSVM
        import joblib
        self.BC_args = add_parser()
        BC_model_path = './BC_module/weights/osvm.pkl'
        self.classifier = joblib.load(BC_model_path)
        self.ID_counter = {}

    def BC_preprocess(self):
        # final shape we should get:
        # video_list shape: [number of video, number of frame, number of IDs]
        # label_list shape: [number of video, 1, number of id]

        from BC_module.data_preprocess import id_normalize, mapping_list
        # calculate total id and make dictionary.
        fake_label_list = {}
        # Constraint ID number
        ID_counter_sorted = dict(sorted(self.ID_counter.items(), key=lambda item: item[1], reverse=True))
        top_k_ID = [k for idx, k in enumerate(ID_counter_sorted) if idx < self.BC_args.id_num]
        sub_frames = []
        for frame in self.traj:
            sub_frame = {}
            for k in frame.keys():
                # only get top k frequencies ID to do GraphRQI
                if k not in fake_label_list and k in top_k_ID:
                    fake_label_list[k] = 0
                    sub_frame[k] = frame[k]
            sub_frames.append(sub_frame)
        if len(self.future_trajs) > 0:
            # print(self.future_trajs.keys(), top_k_ID)
            future_sub_frames = [dict() for lll in range(self.traj_len_required)]
            for k, v in self.future_trajs.items():
                if k in top_k_ID:
                    for f_idx in range(self.traj_len_required):
                        future_sub_frames[f_idx][k] = v[f_idx]

            # make the id number starts from 1
            sub_frames.extend(future_sub_frames)

        tmp_traj, new_fake_label_list, m_list = id_normalize([sub_frames], [[fake_label_list]])
        return (tmp_traj, new_fake_label_list)

    def BC_run(self):

        hint_str = ''
        from BC_module.gRQI_main import computeA, extractLi
        from BC_module.gRQI_custom import RQI
        if len(self.ID_counter) < self.BC_args.id_num:
            # print("No enough ID.")
            return 
        # if futures is empty list that mean don't predict future trajectory
        trajs, labels = self.BC_preprocess()
        adj = computeA(trajs, labels, self.BC_args.neighber, self.BC_args.dataset, True)
        Laplacian_Matrices = extractLi(adj)
        U_Matrices = RQI(Laplacian_Matrices)

        # print("U_Matrices.shape: ",np.shape(U_Matrices))
        new_Matrices = np.reshape(U_Matrices, (-1, self.BC_args.id_num)) 
        # print(new_Matrices.shape)
        res = self.classifier.predict(new_Matrices)
        self.BC_res = res
        # In One Class SVM classification result,
        # -1 mean outlier, 1 mean inlier
        # -1 mean aggressive, 1 mean conservative
        if len(self.future_trajs) == 0:
            hint_str = "Behavior Classification without future Trajectory."
        else:
            hint_str = "predict BC using future trajs"

            # self.traj_reset()
            self.traj_reset_flag = True
            # TO DO: needs to Conbine past traj and future traj. 
    # ========================= OT module code ==========================
    def init_OT(self):
        msg = "Initializing OT Module..."
        prGreen(msg)
        from OT_module.main import inference, set_opt
        self.inference_ptr = inference
        self.OT_args = set_opt()

        # from OT_module.yolact_edge_project.eval import load_yolact_edge
        # self.yolact_edge_model = load_yolact_edge()

    def OT_run(self, frame):
        flag = self.inference_ptr(self.objdet_outputs, frame)
        return flag
    # ========================= Other small function code ==========================
    def show(self, frame):
        bbox = self.MOT_result
        traj_flag = False
        # Draw bounding box & agent ID
        for id in bbox.keys():
            x1, y1, offset_x, offset_y = bbox[id]
            x2, y2 = x1 + offset_x, y1 + offset_y
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
                    thickness=1)
        # Draw trajectory using point
        try:
            # Draw past traj
            for f in self.traj:
                for k, v in f.items():
                    cv2.circle(frame, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
            # Draw future traj
            for k, v in self.future_trajs.items():
                traj_flag = True
                for x, y in v:
                    cv2.circle(frame, (int(x), int(y)), 3, (255,0,0), -1)
            # wk = 1
            # if traj_flag:
            #     wk = 0

            # wk: if future_traj is drawn, then waitkey set 0 to good visualization.
            wk = 0 if traj_flag else 1

            cv2.imshow('t', frame)
            if cv2.waitKey(wk) == 27: # whether is pressed ESC key.
                print("ESC pressed.")
                return True
        except:
            print("Something Wrong... Except Happened!!!!!!!!!!")
