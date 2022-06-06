import os
import logging
import numpy as np
import cv2
from TP_module.util import prGreen

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
        self.init_MOT()
        self.init_TP()
        self.init_BC()
        self.init_OT()

    def init_MOT(self):
        msg = "Initializing MOT Module..."
        # print("Initializing MOT Module...")
        # logging.critical("Initializing MOT Module...")
        # import necessary package
        prGreen(msg)
        from MOT_module.tools.demo_track_yolov5 import make_parser as parser_MOT
        from MOT_module.tracker.byte_tracker import BYTETracker
        from MOT_module.tools.demo_track_yolov5 import load_yolov5

        self.mot_args = parser_MOT().parse_args()
        self.tracker = BYTETracker(self.mot_args, frame_rate=self.mot_args.fps)
        self.object_predictor, self.imgsz, self.names = load_yolov5(rt=True)

    def init_TP(self):
        msg = "Initializing TP Module..."
        prGreen(msg)
        model_path = '/media/rvl/D/Work/fengan/code/Trajectory_Prediction/pytorch_ddpg/output/TrajectoryPrediction-run77(best_model_now)'
        from TP_module.ddpg import DDPG
        from TP_module import env as Env
        from TP_module import parser as TP_parser
        TP_args = TP_parser.get_parser()
        TP_args.actor = 'LSTM'
        self.env = Env.environment(data_select=TP_args.mode, mode='test', args=TP_args)
        self.traj_len_required = self.env.nb_t
        nb_states = self.env.nb_s
        nb_actions = self.env.nb_a
        self.agent = DDPG(nb_states, nb_actions, TP_args)
        self.agent.load_weights(model_path)
        self.agent.is_training = False
        self.agent.eval()
        self.policy = lambda x: self.agent.select_action(x, decay_epsilon=False)
        
    def init_BC(self):
        msg = "Initializing BC Module..."
        prGreen(msg)
        from BC_module.parser import add_parser
        from sklearn.svm import OneClassSVM
        import joblib
        self.BC_opt = add_parser()
        BC_model_path = './BC_module/weights/osvm.pkl'
        self.classifier = joblib.load(BC_model_path)
        self.current_ID = []

    def init_OT(self):
        msg = "Initializing OT Module..."
        prGreen(msg)
        from OT_module.main import inference, set_opt
        from OT_module.yolact_edge_project.eval import load_yolact_edge
        self.inference_ptr = inference
        self.OT_args = set_opt()
        self.yolact_edge_model = load_yolact_edge()

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
        
        return results
    
    def update_traj(self, data):
        frame = {}
        for id in data.keys():
            center_x = data[id][0] + data[id][2]
            center_y = data[id][1] + data[id][3]
            frame[id] = [center_x, center_y]
            if id not in self.current_ID:
                self.current_ID.append(id)
        self.traj.append(frame)
        # print("Update Traj Finished.")

    def show(self, frame, bbox):
        for id in bbox.keys():
            x1, y1, offset_x, offset_y = bbox[id]
            x2, y2 = x1 + offset_x, y1 + offset_y
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
                    thickness=1)
        cv2.imshow('t', frame)
        if cv2.waitKey(0) == 27:
            print("ESC pressed.")            

    def predict_traj(self, traj):
        # read last 5 traj to predict
        init_state = traj[-10:]
        future_traj = []
        current_state = np.array(init_state)
        for _ in range(self.traj_len_required):
            action = self.policy(current_state)
            next_traj = [traj[-2] + action[0], traj[-1] + action[1]]
            future_traj.append(next_traj)
            
            next_state = np.concatenate((current_state[2:], np.asarray(next_traj, dtype=np.float32)), axis=0)
            current_state = next_state
        return future_traj

    def get_future_traj(self):
        if not self.is_enough_tarj():
            return {}
        future_trajs = {}
        for id, traj in self.traj.items():
            # self.traj_len_required mean state number
            # traj is combined x1, y1, x2, y2 ..., xi, yi
            # so self.traj_len_required needs * 2 
            if len(traj) >= self.traj_len_required * 2:
                if id not in future_trajs:
                    future_trajs[id] = []
                future_trajs[id].append(self.predict_traj(traj))
        return future_trajs
    
    def is_enough_tarj(self):
        # one of traj has long enough to predict trjectory.
        for traj in self.traj.values():
            if len(traj) >= self.traj_len_required:
                return True
        return False

    # def is_enough_BC(self):

    def BC_preprocess(self):
        # final shape we should get:
        # video_list shape: [number of video, number of frame, number of IDs]
        # label_list shape: [number of video, 1, number of id]

        from BC_module.data_preprocess import id_normalize, mapping_list
        # calculate total id and make dictionary.
        fake_label_list = {}
        for frame in self.traj:
            # print(frame)
            for k in frame.keys():
                if k not in fake_label_list:
                    fake_label_list[k] = 0
        # make the id number starts from 1
        tmp_traj, new_fake_label_list = id_normalize([self.traj], [[fake_label_list]])
        # print(mapping_list[-1])
        return (tmp_traj, new_fake_label_list)

    def BC_run(self, futures):
        from BC_module.gRQI_main import computeA, extractLi
        from BC_module.gRQI_custom import RQI
        if len(self.current_ID) < self.BC_opt.id_num:
            # print("No enough ID.")
            return 
        # if futures is empty list that mean don't predict future trajectory
        trajs, labels = self.BC_preprocess()
        adj = computeA(trajs, labels, self.BC_opt.neighber, self.BC_opt.dataset, True)
        Laplacian_Matrices = extractLi(adj)
        U_Matrices = RQI(Laplacian_Matrices)
        # print(np.shape(U_Matrices))
        new_Matrices = np.reshape(U_Matrices, (-1, self.BC_opt.id_num)) 
        # print(new_Matrices.shape)
        res = self.classifier.predict(new_Matrices)
        # In One Class SVM classification result,
        # -1 mean outlier, 1 mean inlier
        # -1 mean aggressive, 1 mean conservative
        input()
        if futures == []:
            print("Behavior Classification without future Trajectory.")
        else:
            print("predict BC using future trajs")
    def OT_run(self, frame):
        flag = self.inference_ptr(self.yolact_edge_model, self.objdet_outputs, frame)
        return flag
