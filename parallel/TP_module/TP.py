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
        # self.traj = []
        self.ID_counter = {}
        self.counter = 0
        self.exe_time = 0
        self.single_traj_time = 0
        self.traj_pred_counter = 0
        self.result = None

    def update_ID_counter(self, tmp_dict):
        self.ID_counter = {k:len(v) for k, v in tmp_dict.items()}
        # for k, v in tmp_dict.items():
        #     self.ID_counter[k] = len(v)

    def update_traj(self, data):
        # print(id(traj_share))
        self.current_frame = {}
        for k in data.keys():
            # record ID appear times.
            center_x = data[k][0] + data[k][2] // 2
            center_y = data[k][1] + data[k][3] // 2
            self.current_frame[k] = [center_x, center_y]
            # if k not in self.ID_counter:
            #     self.ID_counter[k] = 0
            # self.ID_counter[k] += 1
        # current_traj_dict[frame_id] = frame

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
        return np.transpose(np.array(future), (1,0,2))
        
    def is_some_id_predictable(self):
        return True in [val > self.traj_len_required for val in self.ID_counter.values()]

    def run(self, current_traj_id_dict, lock):
        total_trajs_id = []
        self.ids = []
        # self.result = None
        t1 = time.time()
        for k, v in self.ID_counter.items():
            if v >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                traj_id = current_traj_id_dict[k]
                total_trajs_id.append(traj_id[-self.traj_len_required:])
                self.ids.append(k)
        if len(total_trajs_id) > 0:
            # tk = time.time()
            self.future = self.predict_traj(total_trajs_id)
            self.result = (dict(zip(self.ids, self.future)))

    def traj_reset(self):
        # self.traj = []
        self.ID_counter = {}
