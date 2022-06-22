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
        self.ID_counter = {}
        self.future_trajs = None
        self.counter = 0
        self.exe_time = 0
        self.single_traj_time = 0
        self.traj_pred_counter = 0

    def update_traj(self, data):
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

    def is_some_id_predictable(self):
        return True in [val > self.traj_len_required for val in self.ID_counter.values()]

    def predict_traj(self, traj):
        # read last k traj to predict
        init_state = traj[-self.traj_len_required:]
        future_traj = [[]] * self.traj_len_required
        # current_state: [x0, y0, x1, y1, .... xk, yk]
        current_state = np.array(init_state).reshape(-1)
        # get future trajectory that length is the same with trajectory history.
        t1 = time.time()
        for idx in range(self.traj_len_required):
            # print(current_state.shape)
            # input()
            action = self.policy(current_state)
            next_traj = [current_state[-2] + action[0], current_state[-1] + action[1]]
            future_traj[idx] = next_traj
            next_state = np.concatenate((current_state[2:], np.asarray(next_traj, dtype=np.float32)), axis=0)
            current_state = next_state
        t2 = time.time()
        # print("TP time for single trajectory:", t2-t1)
        self.single_traj_time += (t2-t1)
        self.traj_pred_counter += 1
        return future_traj

    def run(self):
        st = time.time()
        self.future_trajs = {}
        traj_id = []
        for k, v in self.ID_counter.items():
            if v >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                traj_id = [frame[k] for frame in self.traj if k in frame]
                # for frame in self.traj:
                #     if k in frame:
                #         traj_id.append(frame[k])
                # if id not in future_trajs:
                    # future_trajs[id] = []
                # print(traj_id)
                self.future_trajs[k] = self.predict_traj(traj_id)

        if len(self.future_trajs) == 0:
            self.future_trajs = None

        if self.future_trajs is not None:
            # calculate TP time when executing
            # print("TP time: ", time.time()-st)
            self.counter += 1
            self.exe_time += (time.time() - st)

    def traj_reset(self):
        self.traj = []
        self.ID_counter = {}
        self.future_trajs = None
