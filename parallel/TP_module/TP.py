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
        self.future_trajs = None
        self.counter = 0
        self.exe_time = 0
        self.single_traj_time = 0
        self.traj_pred_counter = 0

    def update_traj(self, data, frame_id, traj_share):
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
        traj_share[frame_id] = frame

    def predict_traj(self, total_trajs_id):
        future = []
        state = np.array(total_trajs_id)
        # state = total_trajs_id[:5,:,:]
        # state = total_trajs_id
        # t1 = time.time()
        for _ in range(self.traj_len_required):
            action = self.policy(state)
            # action = np.reshape(action,(action.shape[0], 1, -1))
            next_traj = state[:,-1,:] + action
            future.append(next_traj)
            next_traj = np.reshape(next_traj, (next_traj.shape[0], 1, -1))
            next_state = np.concatenate((state[:,1:,:], next_traj), axis=1)
            state = next_state
        # future = np.array(future)
        # future = np.transpose(future, (1,0,2))
        # return future
        return np.transpose(np.array(future), (1,0,2))
    def is_some_id_predictable(self):
        return True in [val > self.traj_len_required for val in self.ID_counter.values()]

    def run(self, fm_id, current_traj_dict, future_traj_dict):
        total_trajs_id = []
        ids = []
        for k, v in self.ID_counter.items():
            if v >= self.traj_len_required:
                # collect trajs from buffer(self.traj)
                traj_id = [frame[k] for frame in current_traj_dict.values() if k in frame]
                total_trajs_id.append(traj_id[-self.traj_len_required:])
                ids.append(k)
        # if len(total_trajs_id) > 0:
        future = self.predict_traj(total_trajs_id)
        # print("inference time for batch {}:".format(state.shape[0]), time.time()-t1)
        
        # Update share dict
        future_traj_dict[fm_id] = (dict(zip(ids, future)))
            
    def traj_reset(self):
        # self.traj = []
        self.ID_counter = {}
        self.future_trajs = None
