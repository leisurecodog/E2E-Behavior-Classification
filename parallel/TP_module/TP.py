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

    def update_traj(self, data, traj_share):
        # print(id(traj_share))
        frame = {}
        for k in data.keys():
            # record ID appear times.
            center_x = data[k][0] + data[k][2] // 2
            center_y = data[k][1] + data[k][3] // 2
            frame[k] = [center_x, center_y]
            if k not in self.ID_counter:
                self.ID_counter[k] = 0
            self.ID_counter[k] += 1
        traj_share.append(frame)

    def predict_traj(self, traj):
        # read last k traj to predict
        init_state = traj[-self.traj_len_required:]
        future_traj = [[]] * self.traj_len_required
        # current_state: [x0, y0, x1, y1, .... xk, yk]
        current_state = np.array(init_state).reshape(-1)
        # get future trajectory that length is the same with trajectory history.
        t1 = time.time()
        for idx in range(self.traj_len_required):
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

    def run(self, current_traj_list, future_traj_dict):
        total_trajs_id = []
        ids = []
        future = []
        # print(id(future_traj_dict))
        for k, v in self.ID_counter.items():
            if v >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                traj_id = [frame[k] for frame in current_traj_list if k in frame]
                total_trajs_id.append(traj_id[-self.traj_len_required:])
                ids.append(k)
        if len(total_trajs_id) > 0:
            total_trajs_id = np.array(total_trajs_id)
            # state = total_trajs_id[:5,:,:]
            state = total_trajs_id
            t1 = time.time()
            for _ in range(self.traj_len_required):
                action = self.policy(state)
                # action = np.reshape(action,(action.shape[0], 1, -1))
                next_traj = state[:,-1,:] + action
                future.append(next_traj)
                next_traj = np.reshape(next_traj, (next_traj.shape[0], 1, -1))
                next_state = np.concatenate((state[:,1:,:], next_traj), axis=1)
                state = next_state
            print("inference time for batch {}:".format(state.shape[0]), time.time()-t1)
            # update dict without assign a new object
            future_traj_dict.update(dict(zip(ids, future)))
        # print(id(future_traj_dict))
    def traj_reset(self):
        self.traj = []
        self.ID_counter = {}
        self.future_trajs = None