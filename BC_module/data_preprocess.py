import os

mapping_list = []

def pre_calculate(label_path, mot_path, file): # read one video's aggressive_list & appear_id
    file_path = os.path.join(label_path, file)
    aggressive_list = [] # save each aggressive id
    aggressive_dic = {} # save appear frames for each aggressive id.
    appear_id = {} # save each id appear times in aggressive_dict
    # convert label to dict
    with open(file_path, 'r') as f:
        tmp = f.readlines()
        for t in tmp:
            tl = t.split(' ')
            k, v = int(tl[0]), int(tl[1])
            if v == 1 and k not in aggressive_list:
                aggressive_list.append(k)
    for id in aggressive_list:
        aggressive_dic[id] = []
    
    # step 1 & step 2
    d = file.replace('.txt', '')
    d_path = os.path.join(mot_path, d)
    txts = sorted(os.listdir(d_path))
    
    for idx, txt in enumerate(txts): # read each txt file
        flag = False
        p = os.path.join(d_path, txt)
        with open(p, 'r') as f:
            tmp = f.readlines()
            for t in tmp:
                tl = [float(data) for data in t.split(' ')] # class id cen_x/w cen_y/h tw/w th/h
                if int(tl[1]) in aggressive_list:
                    aggressive_dic[int(tl[1])].append(idx)
                    flag = True
            if flag: # aggressive id is found in this frame, record all id in this frame.
                for t in tmp:
                    tl = [float(data) for data in t.split(' ')]
                    if int(tl[1]) not in aggressive_list:
                        if int(tl[1]) not in appear_id:
                            appear_id[int(tl[1])] = 1
                        else:
                            appear_id[int(tl[1])] += 1
    sorted_appear_id = {k:appear_id[k] for k in sorted(appear_id, key=appear_id.get, reverse=True)}

    return aggressive_list, sorted_appear_id

def extract_id_before_KNN(label_path, mot_path, file, opt):
    '''
    procedure  CREATE LABEL:
    step 1. search the frames where the aggressive in and use list to store.
    step 2. calculate the ids in same frame.
    step 3. select the top n1-th id according the step 2.
    step 4. select top top n2-th id according the euclidean distance to the aggressive id.
    step 5. normalize the id
    '''
    aggressive_list, appear_id = pre_calculate(label_path, mot_path, file)
    
    # step 3: select the top n Frequent id
    if len(appear_id) >= opt.id_num and opt.ext_mode == 1:
        new_dict = {k:appear_id[k] for k in sorted(appear_id, key=appear_id.get, reverse=True)}
        final_dict = {k:1 for idx, k in enumerate(aggressive_list) if idx < opt.id_num} 
        #constraints that prevent aggressive id more than opt.id_num
        remain_id_num = opt.id_num - len(final_dict)
        remain_id_dict = {k:0 for i, (k, v) in enumerate(new_dict.items()) if i < remain_id_num}
        final_dict.update(remain_id_dict)
        return [final_dict]
    elif opt.ext_mode == 2:
        return aggressive_list, appear_id

def get_total_label(label_path, file, opt):
    dics = {}
    p = os.path.join(label_path, file)
    with open(p, 'r') as f:
        tmp = f.readlines()
        for t in tmp:
            tl = t.split(' ')
            dics[int(tl[0])] = int(tl[1])
    return [dics]

def get_clips_label(label_path, mot_path, file, opt): # using by extract mode 4
    clip_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k/bdd100k/aggressive_clips'
    labels_file = []
    file_clips = []
    file_path = os.path.join(clip_path, file.replace('.txt','_clips.txt'))
    dics = {}
    label_loc = os.path.join(label_path, file)
    with open(label_loc, 'r') as f:
        tmp = f.readlines()
        for t in tmp:
            tl = t.split(' ')
            dics[int(tl[0])] = int(tl[1])
    with open(file_path, 'r') as f:
        tmp = f.readlines()
        for idx, t in enumerate(tmp):
            labels_file.append([dics])
            file_clips.append(file.replace('.txt', '_{}'.format(idx)))
    return labels_file, file_clips

func_ptr = {1: extract_id_before_KNN}

def get_labels(label_path, mot_path, opt):

    files = os.listdir(label_path)
    labels_list = []
    cal = 0
    file_list = []
    aggressive_list = []
    appear_id = []
    file_clips = []
    for i, file in enumerate(files):
        if i == opt.vid_num:
            break
        res_list = []
        if opt.ext_mode == 1:
            res_list = func_ptr[opt.ext_mode](label_path, mot_path, file, opt)
        elif opt.ext_mode == 2: # get all id labels & do post extract id
            res_list = get_total_label(label_path, file, opt)
            aa, bb = func_ptr[1](label_path, mot_path, file, opt)
            aggressive_list.append(aa)
            appear_id.append(bb)
        elif opt.ext_mode == 4:
            res_list, file_clips = get_clips_label(label_path, mot_path, file, opt)

        if res_list is not None:
            cal += 1
            # print(len(labels_list))
            if opt.ext_mode == 4:
                labels_list.extend(res_list)
                file_list.extend(file_clips)
                # print(labels_list) 
            else:
                labels_list.append(res_list)
                file_list.append(file)

    print("{}/{}".format(cal, len(files)))
    # print(labels_list)
    return labels_list, file_list, aggressive_list, appear_id

        
def get_datas(mot_path, labels_list, file_list, opt):
    videos_list = []
    # print(labels_list)
    import cv2
    for i, file in enumerate(file_list):
        if opt.dataset == 'meteor':
            vid_path = os.path.join('/home/fengan/Desktop/Dataset/METEOR/videos', file.replace('.txt','.MP4'))
            cap = cv2.VideoCapture(vid_path) 
        video = []
        dir_name = file.replace('.txt', '')
        if opt.ext_mode == 4:
            dir_name = dir_name.split('_')[0]
        
        dir_loc = os.path.join(mot_path, dir_name)
        txts = []
        # print(dir_loc)
        if opt.ext_mode == 4:
            clip_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k/bdd100k/aggressive_clips'
            line = int(file.split('_')[1])
            with open(os.path.join(clip_path, dir_name + '_clips.txt'), 'r') as f:
                tmp = f.readlines()[line]
                tl = tmp.split(' ')
                r0, r1 = tl[0], tl[1]
                # print(r0, r1)
                for j in range(int(r0)+1, int(r1)+2):
                    str_i = dir_name + '-' + str(j).zfill(7) + '.txt'
                    # print(str_i)
                    txts.append(str_i)
        else:
            txts = sorted(os.listdir(dir_loc))
        for idx, txt in enumerate(txts):
            if idx % opt.gap != 0:
                continue
            d = os.path.join(dir_loc, txt)
            if opt.dataset == 'own':
                img_loc = d.replace('labels_with_ids_offset', 'images').replace('txt','jpg')
                # print(img_loc)
                img = cv2.imread(img_loc)
            elif opt.dataset == 'meteor':
                
                ret, img = cap.read()
            shape = img.shape
            frame = {}            
            with open(d, 'r') as f:
                tmp = f.readlines()
                for t in tmp:
                    tl = [float(data) for data in t.split(' ')]

                    if int(tl[1]) not in labels_list[i][0].keys():
                        continue
                    frame[int(tl[1])] = [int(tl[2]*shape[1]), int(tl[3]*shape[0])]
            video.append(frame)
        videos_list.append(video)
    return videos_list

def id_normalize(vid_list, l_list):
    global mapping_list
    new_labels_list = []
    new_vid_list = []
    import random
    for i in range(len(vid_list)):
        keys = list(l_list[i][0].keys())
        random.shuffle(keys)
        dic1 = {keys.index(key):l_list[i][0][key]for key in keys}
        new_labels_list.append([dic1])
        vid_dict = []
        for frame in vid_list[i]:
            vid_dict.append({keys.index(key):frame[key] for key in list(frame.keys())})
        new_vid_list.append(vid_dict)
        mapping_list.append({keys.index(key):key for key in keys})
        # print(mapping_list[0], vid_dict[0], dic1)
    return new_vid_list, new_labels_list, mapping_list

def label_clean(videos_list, labels_list, opt):
    """
    original label list is contains all frame's label,
    but this function is working for ext_mode 4,
    so it will extract the labels only exist in aggressive behavior happens.
    """
    new_labels_list = []

    for idx, vids in enumerate(videos_list):
        vid_id = []
        for vid in vids:
            for key in vid.keys():
                if key not in vid_id:
                    vid_id.append(key)
        new_dic = {}
        for k in vid_id:
            new_dic[k] = labels_list[idx][0][k]
        new_labels_list.append([new_dic])
    return new_labels_list

def update_data(videos_list, labels_list, opt):
    # extract id before KNN
    final_vids = []
    final_labels = []
    for idx, vids in enumerate(videos_list):
        id_dict = {}
        
        final_label = {key:1 for key in labels_list[idx][0].keys() if labels_list[idx][0][key] == 1}
        # print(final_label)
        for vid in vids:
            for key in vid.keys():
                if key not in id_dict:
                    id_dict[key] = 1
                else:
                    id_dict[key] += 1
        if len(id_dict) < opt.id_num:
            continue
        id_dict_sorted = {k:id_dict[k] for k in sorted(id_dict, key=id_dict.get, reverse=True)}

        lens = opt.id_num - len(final_label)
        idx = 0
        for id in id_dict_sorted.keys():
            if id not in final_label.keys():
                idx += 1
                final_label[id] = 0
            if idx == lens:
                break
        # print(final_label)
        final_labels.append([final_label])
        final_vid = []
        for vid in vids:
            tmp_dic = {key:vid[key] for key in vid.keys() if key in final_label.keys()}
            # print(tmp_dic)
            final_vid.append(tmp_dic)
        final_vids.append(final_vid)

    return final_vids, final_labels


def create_data_and_label(mot_path, label_path, opt): # new function 
    labels_list, file_list, aggressive_id, appear_id = get_labels(label_path, mot_path, opt)
    
    videos_list = get_datas(mot_path, labels_list, file_list, opt)
    
    print(opt.dataset) 
    if opt.ext_mode == 4:
        labels_list = label_clean(videos_list, labels_list, opt)
        videos_list, labels_list = update_data(videos_list, labels_list, opt)
        
    videos_list, labels_list = id_normalize(videos_list, labels_list)
    
    return videos_list, labels_list, aggressive_id, appear_id

def create_label_and_data(mot_path, label_path, opt): # old function
    create_label_and_data.data_num = opt.vid_num # how many video will be added to train
    
    import random
    import cv2
    files = os.listdir(label_path)
    labels_list = []
    videos_list = []
    tmp = create_label_and_data.data_num
    # create labels
    for file in files:
        tmp -= 1
        if tmp < 0:
            break
        labels = []
        file_path = os.path.join(label_path, file)
        with open(file_path, 'r') as f:
            lst = f.readlines()
            counter = opt.id_num # pick number

            idx_lst = np.arange(len(lst),dtype=int)
            for i in range(len(lst)):
                behavior_label = lst[i].split(' ')[1]
                if int(behavior_label) == 1:
                    idx_lst = idx_lst[idx_lst != i]
                    counter -= 1
                    labels.append(i)
            labels += random.sample(list(idx_lst), counter)
            ids = [l.split(' ')[0] for l in lst]
            labels = [int(ids[idx]) for idx in labels]
            label_dict = {}
            
            aggressive_counter = opt.id_num

            for l in labels:
                if aggressive_counter != counter:
                    aggressive_counter -= 1
                    label_dict[l] = 1
                else:
                    label_dict[l] = 0

            labels_list.append([label_dict])
    # create data
    for idx, file in enumerate(files):
        print(file)
        create_label_and_data.data_num -= 1
        if create_label_and_data.data_num < 0:
            break
        f_name = file.split('.')[0]
        label = labels_list[idx]
        video = []
        mot_file_data = os.path.join(mot_path, f_name)
        for i in sorted(os.listdir(mot_file_data)):
            loc = os.path.join(mot_file_data, i)
            img_loc = loc.replace('labels_with_ids_offset', 'images').replace('txt','jpg')
            img = cv2.imread(img_loc)
            shape = img.shape
            frames = {}
            with open(loc, 'r') as f:
                lst = f.readlines()
                for l in lst:
                    ll = l.split(' ')
                    if int(ll[1]) not in label[0]:
                        continue
                    frames[int(ll[1])] = [int(float(ll[2])*shape[1]), int(float(ll[3])*shape[0])]
            video.append(frames)
            # print(video)
        videos_list.append(video)
            
    return videos_list, labels_list

def post_extract_id(matrices, aggressive_id, appear_id, labels_list, opt):
    """
    using for extract mode 2
    it will extract id after the graphRQI done and before into the ML or DL model.
    the id number is decided by opt.id_num
    """
    print(len(matrices), len(aggressive_id), len(appear_id))
    post_U_Metrices = []
    labels = []
    for idx, u in enumerate(matrices):
        if len(aggressive_id[idx]) + len(appear_id[idx]) < opt.id_num:
            continue
        select_id = []
        for id in aggressive_id[idx]:
            select_id.append(id)
        remain_id_num = opt.id_num - len(aggressive_id[idx])

        for idk, id in enumerate(appear_id[idx].keys()):
            if idk == remain_id_num:
                break
            select_id.append(id)
        # extract id
        print(select_id)
        np_arr = np.array(u)
        n_arr = np_arr[:, select_id]
        final_metrix = n_arr[select_id,:]
        post_U_Metrices.append(final_metrix)
        label_new = {}
        for l in labels_list[idx][0].keys():
            if l in select_id:
                label_new[l] = labels_list[idx][0][l]
            
        labels.append([label_new])
        # print(label_new)
    return post_U_Metrices, labels
