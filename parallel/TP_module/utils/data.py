import os
# from this import d
from tqdm import tqdm
import cv2
import numpy as np
import torch.utils.data as Data
import torch
# from .rl import find_expert_SVF_1d

def bdd100k_process(loc, wh=False): 
    num_vid = 1000
    label_path = os.path.join(loc, 'labels_with_ids/track/train')
    folders = os.listdir(label_path)
    dict_list = []
    hw_list = []
    h = 0
    w = 0
    for idx, folder in enumerate(tqdm(folders)):
        if idx == num_vid:
            break
        dics = {}
        dict_wh = {}
        folder_path = os.path.join(label_path, folder)
        files = os.listdir(folder_path)
        files = sorted(files)
        for idx, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            image_path = file_path.replace('labels_with_ids', 'images').replace('.txt','.jpg')
            if idx == 0:
                h, w = cv2.imread(image_path).shape[:-1]
            with open(file_path, 'r') as f:
                tmp = f.readlines()
                for tl in tmp:
                    t_list = tl.split(' ')
                    if int(t_list[1]) not in dics:
                        dics[int(t_list[1])] = []
                    dics[int(t_list[1])].append([float(t_list[2])*w, float(t_list[3])*h])
        dict_list.append(dics)
        hw_list.append([h, w])
    if wh == False:
        return dict_list
    return dict_list, hw_list
    

def meteor_process(loc):
    label_path = os.path.join(loc, 'MOT labels_with_ids')
    folders = os.listdir(label_path)
    dict_list = []
    height = 0
    width = 0
    for idx, folder in enumerate(tqdm(folders)):
        if idx == 20:
            break
        dics = {}
        folder_path = os.path.join(label_path, folder)
        files = os.listdir(folder_path)
        for idx, file in enumerate(sorted(files)):
            file_path = os.path.join(folder_path, file)
            # image_path = file_path.replace('labels_with_ids', 'images').replace('.txt','.jpg')
            # if idx == 0:
            #     height, width = cv2.imread(image_path).shape[:-1]
            with open(file_path, 'r') as f:
                tmp = f.readlines()
                for tl in tmp:
                    t_list = tl.split(' ')
                    if int(t_list[1]) not in dics:
                        dics[int(t_list[1])] = []
                    dics[int(t_list[1])].append([float(t_list[2])*119.0, float(t_list[3])*119.0])
        dict_list.append(dics)
    # print(dict_list[0])
    return dict_list

def outside_process(opt):
    import utils.offroad_loader as offroad_loader
    from torch.utils.data import DataLoader

    train_loader = offroad_loader.OffroadLoader(grid_size=80, tangent=False, more_kinematic=0.3)
    train_loader = DataLoader(train_loader, num_workers=1, batch_size=1, shuffle=True)
    p = []
    f = []
    for idx, (feat, past_traj, future_traj) in enumerate(train_loader):
        # print(idx)
        p.append(past_traj[0].detach().numpy())
        f.append(future_traj[0].detach().numpy())
    p = torch.Tensor(p)
    f = torch.Tensor(f)
    dataset = Data.TensorDataset(p ,f)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )
    return p, f, loader

def bdd100k_label_full_process(loc):
    folders = os.listdir(loc)
    dict_list = []
    for folder_idx, folder in enumerate(folders):
        dics = {}
        id_list = []
        if folder_idx == 600:
            break
        label_path = os.path.join(loc, folder)
        frame_idx = 1
        tmp_list = []
        with open(label_path, 'r') as f:
            tmp_list = f.readlines()

        frame_data_dic = {}
        for line in tmp_list: # split whole data into each frame
            id = line.split(',')[1]
            frame_id = line.split(',')[0]
            if frame_id not in frame_data_dic:
                frame_data_dic[frame_id] = []
            if id not in id_list:
                id_list.append(int(id))
            frame_data_dic[frame_id].append(line)
        # handle trajectory
        for k, v in frame_data_dic.items():
            appear_id_list = []
            for i in range(len(v)):
                data = v[i].split(',')
                center_x = float(data[2]) + float(data[4]) / 2
                center_y = float(data[3]) + float(data[5]) / 2
                if int(data[1]) not in dics:
                    dics[int(data[1])] = []
                dics[int(data[1])].append([center_x, center_y])
                appear_id_list.append(int(data[1]))
            
            for IDs in id_list:
                if IDs not in dics:
                    dics[IDs] = []
                if IDs not in appear_id_list:
                    dics[IDs].append([np.inf, np.inf])
        
        for k, v in dics.items():
            longest_st_idx = -1
            longest_end_idx = -1
            st_idx = -1
            end_idx = -1
            flag = False
            # print(dics[k])
            for i in range(len(v)):
                if v[i][0] != np.inf and flag == False:
                    st_idx = i
                    flag = True
                if v[i][0] == np.inf and flag == True:
                    end_idx = i-1
                    flag = False
                    if abs(end_idx - st_idx) > abs(longest_end_idx - longest_st_idx):
                        longest_end_idx = end_idx
                        longest_st_idx = st_idx
            dics[k] = v[longest_st_idx:longest_end_idx]
            # print(dics[k])
            # input()
                
        # for line in tmp_list:
        #     data = line.split(',')
        #     if int(data[1]) not in dics:
        #         dics[int(data[1])] = []
        #     center_x = float(data[2]) + float(data[4]) / 2
        #     center_y = float(data[3]) + float(data[5]) / 2
        #     dics[int(data[1])].append([center_x, center_y])
            
        dict_list.append(dics)
    return dict_list

def kitti_tracking_process(loc):
    label_convert = {'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3, 'Person':3, 'Cyclist':4, 'Tram':5, 'Misc':6, "DontCare":6}
    folders = os.listdir(loc)
    dict_list = []
    for folder in folders:
        dics = {}
        tmp_list = []
        label_path = os.path.join(loc, folder)
        with open(label_path, 'r') as f:
            tmp_list = f.readlines()
        for line in tmp_list:
            data = line.split(' ')
            if label_convert[data[2]] == 6:
                continue
            if int(data[1]) not in dics:
                dics[int(data[1])] = []
            center_x = (float(data[6]) + float(data[8])) / 2
            center_y = (float(data[7]) + float(data[9])) / 2
            dics[int(data[1])].append([center_x, center_y])
        dict_list.append(dics)

    return dict_list

def data_preprocess(dataset):
    # ApolloScape data format:
    # frame_number obj_id obj_type x y z obj_len obj_wei obj_hei heading
    dict_list = []
    if dataset == 'apolloScape':
        train_data_path = '/home/fengan/Desktop/Dataset/ApolloScape/Trajectory/prediction_train'
        test_data_path = '/home/fengan/Desktop/Dataset/ApolloScape/Trajectory/prediction_test'
        # process the training data
        train_files = os.listdir(train_data_path)
        for file_name in sorted(train_files):
            dics = dict()
            file_path = os.path.join(train_data_path, file_name)
            with open(file_path, 'r') as f:
                lst = f.readlines()
                for line in lst:
                    line_split = line.split(' ')
                    if int(line_split[1]) not in dics:
                        dics[int(line_split[1])] = []
                    dics[int(line_split[1])].append([float(line_split[3]), float(line_split[4])])
            # print(dics)
            dict_list.append(dics)
        # return dict_list
    elif dataset == 'bdd100k':
        train_data_path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/bdd100k/bdd100k'
        dict_list = bdd100k_process(train_data_path)
    elif dataset == 'bdd100k_ano_half':
        train_data_path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/bdd100k_ano_half'
        dict_list = bdd100k_process(train_data_path)
    elif dataset == 'meteor':
        train_data_path = '/media/rvl/D/Work/fengan/Dataset/METEOR'
        dict_list = meteor_process(train_data_path)
    elif dataset == 'bdd100k_full':
        train_data_path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/label/bdd100k_box_track_20_labels_trainval/bdd100k/labels/box_track_20/txt_folder'
        dict_list = bdd100k_label_full_process(train_data_path)
    elif dataset == 'kitti':
        train_data_path = '/media/rvl/D/Work/fengan/Dataset/KITTI Tracking/data_tracking_label_2/training/label_02'
        dict_list = kitti_tracking_process(train_data_path)
    
    return dict_list

def train_dataloader(datas, opt, shuffle, eval=False):
    srcs = []
    targets = []
    for data in datas:
        for id, traj in data.items():
            # print(id, traj)
            if len(traj) < opt.obs_len * 2:
                continue
            srcs.append(traj[:opt.obs_len])
            targets.append(traj[opt.obs_len:opt.obs_len*2])
    srcs = torch.Tensor(srcs)
    targets = torch.Tensor(targets)
    dataset = Data.TensorDataset(srcs, targets)
    size = opt.batch_size
    if eval == True:
        size = 1
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = size,
        shuffle = shuffle,
        num_workers = opt.num_workers
    )
    return srcs, targets, loader
# ==================================feature prepare stage==============================

def prepare_feature(model, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = 'weights/best.pt'
    
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    if opt.feature_dataset == 'outside':
        past_traj, future_traj, dataloader = outside_process(opt)
    else:
        trajectories = data_preprocess(opt.feature_dataset)
        past_traj, future_traj, dataloader = train_dataloader(trajectories, opt, shuffle=False, eval=True)
    h_list = []
    final_h = ''
    with torch.no_grad():
        for s, t in tqdm(dataloader):
            src = s.to(device)
            trg = t.to(device)
            # we only want to the hidden states for all time t
            output, hidden = model(src, trg)
            hidden = torch.unsqueeze(hidden, 0)
            
            h_list.append(hidden)

    # concatenate each data of seqence
    for i in range(len(h_list)):
        if i == 0:
            final_h = h_list[i]
        else:
            final_h = torch.cat([final_h, h_list[i]])
    
    return final_h, past_traj, future_traj

def fill_feature_map(traj, feature_data, vtp, opt):
    f_len = opt.f_len
    feature_data = feature_data.cpu().detach().numpy()
    
    fm_list = []
    fm = np.zeros((f_len, vtp.grid_size, vtp.grid_size))
    for i in range(len(traj)):
        tmp_traj = traj[i].detach().numpy()
        tmp_traj = np.array([tmp_traj[~np.isnan(tmp_traj).any(axis=1)]])
        # fm = np.zeros((1, vtp.grid_size, vtp.grid_size))
        for j in range(tmp_traj.shape[1]):
            x, y = round(float(traj[i][j][0])), round(float(traj[i][j][1]))
            # fm[0,y,x] += np.sum(feature_data[i,j,:])
            fm[:,y,x] += np.sum(feature_data[i,0,j,:])
        if i % opt.map_batch == 0 and i: # every opt.map_batch traj combine into a feature map
            fm_list.append(fm)
            fm = np.zeros((f_len, vtp.grid_size, vtp.grid_size))
    
    return np.array(fm_list)

def feature_dataloader(past_traj, future_traj, feature_data, vtp, opt):
    feature_maps = fill_feature_map(past_traj, feature_data, vtp, opt)
    new_future_traj = []
    tmp_arr = []
    for i in range(past_traj.shape[0]):
        if i % opt.map_batch == 0 and i != 0:
            new_future_traj.append(np.array(tmp_arr))
            tmp_arr = []
        tmp_arr.append(future_traj[i].numpy())
    
    future_traj = np.array(new_future_traj)
    feature_maps = torch.Tensor(feature_maps)
    for i in range(len(future_traj)):
        print(future_traj[i].shape)
    future_traj = torch.Tensor(np.array(future_traj))
    # print(feature_maps.shape, future_traj.shape)
    dataset = Data.TensorDataset(feature_maps, future_traj)
    # print(len(dataset))
    if opt.feature_dataset == 'outside':
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.map_batch,
            shuffle = False,
            num_workers = opt.num_workers
        ), None
    else:
        train_len = int(len(dataset) * 0.9)
        test_len = len(dataset) - train_len
        train_dataset, test_dataset = Data.random_split(dataset, [train_len, test_len])

        return Data.DataLoader(
            dataset = train_dataset,
            batch_size = opt.map_batch,
            shuffle = False,
            num_workers = opt.num_workers
        ), Data.DataLoader(
            dataset = test_dataset,
            batch_size = opt.map_batch,
            shuffle = False,
            num_workers = opt.num_workers
        )
