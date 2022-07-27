from TP_module.util import prGreen
import time
import numpy as np

class TP:
    def __init__(self):
        msg = "Initializing TP Module..."
        prGreen(msg)
        from TP_module.ddpg import DDPG
        from TP_module import env as Env
        from TP_module import parser as TP_parser
        TP_args = TP_parser.get_parser()
        TP_args.actor = "seq2seq"
        # TP_args.actor = 'LSTM'
        # TP_args.actor = 'aseq'
        print(TP_args.actor)
        model_path = './TP_module/weights/' + TP_args.actor
        self.env = Env.environment(data_select=TP_args.mode, mode='test', args=TP_args)
        self.traj_len_required = self.env.nb_t 
        nb_states = self.env.nb_s
        nb_actions = self.env.nb_a
        self.agent = DDPG(nb_states, nb_actions, TP_args)
        self.agent.load_weights(model_path)
        self.agent.is_training = False
        self.agent.eval()
        self.policy = lambda x: self.agent.select_action(x, decay_epsilon=False)
        self.traj = []
        self.result = None
        self.traj_id_dict = {}
        self.ID_counter = {}
        self.current_frame_ID_counter = []
        #######################################
        self.counter = 0
        self.exe_time = 0
        self.single_traj_time = 0
        self.traj_pred_counter = 0

    def update_traj(self, data):
        frame = {}
        self.current_frame_ID_counter = {}
        for ID in data.keys():
            # record ID appear times.
            center_x = data[ID][0] + data[ID][2] // 2
            center_y = data[ID][1] + data[ID][3] // 2
            frame[ID] = [center_x, center_y]
            if ID not in self.traj_id_dict:
                self.traj_id_dict[ID] = []
            self.traj_id_dict[ID].append(frame[ID])

            if ID not in self.ID_counter:
                self.ID_counter[ID] = 0
            self.ID_counter[ID] += 1
            self.current_frame_ID_counter[ID] = self.ID_counter[ID] 
        self.traj.append(frame)

    def is_some_id_predictable(self):
        return True in [val > self.traj_len_required for val in self.current_frame_ID_counter.values()]

    # def predict_traj(self, traj):
    #     # read last k traj to predict
    #     init_state = traj[-self.traj_len_required:]
    #     future_traj = [[]] * self.traj_len_required
    #     # current_state: [x0, y0, x1, y1, .... xk, yk]
    #     current_state = np.array(init_state).reshape(-1)
    #     # get future trajectory that length is the same with trajectory history.
    #     t1 = time.time()
    #     for idx in range(self.traj_len_required):
    #         # print(current_state.shape)
    #         # input()
    #         action = self.policy(current_state)
    #         next_traj = [current_state[-2] + action[0], current_state[-1] + action[1]]
    #         future_traj[idx] = next_traj
    #         next_state = np.concatenate((current_state[2:], np.asarray(next_traj, dtype=np.float32)), axis=0)
    #         current_state = next_state
    #     t2 = time.time()
    #     # print("TP time for single trajectory:", t2-t1)
    #     self.single_traj_time += (t2-t1)
    #     self.traj_pred_counter += 1
    #     return future_traj
    def predict_traj(self, total_trajs_id):
        future = []
        state = np.array(total_trajs_id)
        for _ in range(self.traj_len_required):
            
            action = self.policy(state) 
            next_traj = state[:,-1,:] + action
            future.append(next_traj)
            next_traj = np.reshape(next_traj, (next_traj.shape[0], 1, -1))
            next_state = np.concatenate((state[:,1:,:], next_traj), axis=1)
            state = next_state
        return np.transpose(np.array(future), (1, 0, 2))

    def run(self):
        # st = time.time()
        # self.result = {}
        # traj_id = []
        # for k, v in self.current_frame_ID_counter.items():
        #     if v >= self.traj_len_required:
        #         # collect trajs from buffer(self.traj)
        #         traj_id = [frame[k] for frame in self.traj if k in frame]
        #         self.result[k] = self.predict_traj(traj_id)

        # if len(self.result) == 0:
        #     self.result = None

        # if self.result is not None:
        #     self.counter += 1
        #     self.exe_time += (time.time() - st)
        #     print("TPTPTPTPTPTP\t", time.time() - st)
        total_trajs_id = []
        self.ids = []
        self.result = {}
        for ID, T in self.current_frame_ID_counter.items():
            if T >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                total_trajs_id.append(self.traj_id_dict[ID][-self.traj_len_required:])
                self.ids.append(ID)

        if len(total_trajs_id) > 0:
            self.future = self.predict_traj(total_trajs_id)
            self.result = {self.ids[i]:self.future[i].tolist() for i in range(len(self.ids))}

    def traj_reset(self):
        self.traj = []
        self.ID_counter = {}
        self.future_trajs = None
