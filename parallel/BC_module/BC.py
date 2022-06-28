from TP_module.util import prGreen
import time
import numpy as np


class BC:
    def __init__(self, traj_len_required):
        msg = "Initializing BC Module..."
        prGreen(msg)
        from BC_module.parser import add_parser
        from sklearn.svm import OneClassSVM
        import joblib
        self.BC_args = add_parser()
        BC_model_path = './BC_module/weights/osvm.pkl'
        self.classifier = joblib.load(BC_model_path)
        self.traj_len_required = traj_len_required
        self.mapping_list = None
        self.result = None
        self.BC_required_len = 10
        self.counter = 0
        self.exe_time = 0
        # self.bbox_color = [(255, 255, 255), (0, 0, 255)]

    def is_satisfacation(self, id_counter):
        if len(id_counter) < self.BC_args.id_num or max(id_counter.values()) < self.BC_required_len:
            return False
        self.id_counter = id_counter
        return True

    def preprocess(self, current, future):
        # print(future)
        # final shape we should get:
        # video_list shape: [number of video, number of frame, number of IDs]
        # label_list shape: [number of video, 1, number of id]
        from BC_module.data_preprocess import id_normalize, mapping_list
        # calculate total id and make dictionary.
        fake_label_list = {}
        # Constraint ID number
        sub_frames = []
        for frame in current:
            sub_frame = {}
            for k in frame.keys():
                # only get top k frequencies ID to do GraphRQI
                if k not in fake_label_list and k in self.top_k_ID:
                    fake_label_list[k] = 0
                    sub_frame[k] = frame[k]
            sub_frames.append(sub_frame)
        
        if len(future) > 0:
            future_sub_frames = [dict() for lll in range(self.traj_len_required)]
            for k, v in future.items():
                if k in self.top_k_ID:
                    for f_idx in range(self.traj_len_required):
                        future_sub_frames[f_idx][k] = v[f_idx]
            # make the id number starts from 1
            sub_frames.extend(future_sub_frames)
        tmp_traj, new_fake_label_list, self.mapping_list = id_normalize([sub_frames], [[fake_label_list]])
        return (tmp_traj, new_fake_label_list)

    def run(self, current_traj, future_traj):
        # hint_str = "Behavior Classification without future Trajectory."
        from BC_module.gRQI_main import computeA, extractLi
        from BC_module.gRQI_custom import RQI
        # hint_str = "predict BC using future trajs"
        # if futures is empty list that mean don't predict future trajectory
        ID_counter_sorted = dict(sorted(self.id_counter.items(), key=lambda item: item[1], reverse=True))
        self.top_k_ID = [k for idx, k in enumerate(ID_counter_sorted) if idx < self.BC_args.id_num]
        trajs, labels = self.preprocess(current_traj, future_traj)
        st1 = time.time()
        adj = computeA(trajs, labels, self.BC_args.neighber, self.BC_args.dataset, True)
        st2 = time.time()
        Laplacian_Matrices = extractLi(adj)
        st3 = time.time()
        U_Matrices = RQI(Laplacian_Matrices)
        st4 = time.time()
        new_Matrices = np.reshape(U_Matrices, (-1, self.BC_args.id_num)) 
        res = self.classifier.predict(new_Matrices)
        # In One Class SVM classification result,
        # -1 mean outlier, 1 mean inlier => -1 mean aggressive, 1 mean conservative
        # create dict for {id:bc_result}
        self.result = {}
        for k, v in self.mapping_list.items():
            self.result[v] = res[k]
        
        # print("BC time: ", st2-st, time.time()-st2)
        print("??????", st2-st1, st3-st2, st4-st3)
        self.counter += 1
        self.exe_time += (time.time() - st1)