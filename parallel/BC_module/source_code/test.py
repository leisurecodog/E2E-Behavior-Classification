def test():
    dataset = 'traf'
    nbrs = 10
    data_loc = '/home/fengan/Desktop/ByteTrack/YOLOv5_outputs/track_vis/2022_01_25_13_44_48.txt'
    with open(data_loc, 'r') as f:
        label_list = {}
        labels_list = []
        tmp = f.readline()
        frame_id = 1
        video_list =[]
        video = []
        dic = {}
        while(tmp): # read viddeo
            tmp_list = tmp.split(',')
            tmp_list = [float(x) for x in tmp_list]
            if(int(tmp_list[0]) > frame_id):
                frame_id += 1
                video.append(dic)
                dic = {}
            id = int(tmp_list[1])
            dic[id] = [(int(tmp_list[2])+ int(tmp_list[4])) // 2, (int(tmp_list[3])+ int(tmp_list[5])) // 2]
            if id not in label_list.keys():
                label_list[id] = 0 #default behavior
            tmp = f.readline()
            if frame_id >= 1000:
                break
        labels_list.append([label_list])
        video_list.append(video)

        Adjacency_Matrices = computeA ( video_list , labels_list , nbrs , dataset, True)
        # List of lists: Each element of the list corresponds to a list of [L_1,L_2,...,L_T] for each TRAF video
        Laplacian_Matrices = extractLi ( Adjacency_Matrices )

    # ============================================ data convert stage ====================================================
    U_Matrices = []
    from scipy import linalg as LA
    for Lis_for_each_video in Laplacian_Matrices:
        time_start_all = time.time()
        # ListofUs = []
        for L_index, L in enumerate(Lis_for_each_video):
            # print(L)
            # input()
            if first_laplacian(L_index):
                Lambda_prev, U_prev = la.eig(L) # need top k eigenvectors
                # print(U_prev)
                Lambda_prev = np.diag(np.real(Lambda_prev))
                U_prev = np.real(U_prev)
                # print(U_prev)
                
            else:
                # print(U_prev)
                U_curr, Lambda = GraphRQI(U_prev, Lis_for_each_video, L_index, Lambda_prev)
                Lambda_prev = Lambda
                # ListofUs.append(U_curr)
                U_prev = U_curr[-1]
            # Daeig , Va , X = LA.svd ( L , lapack_driver='gesvd' )
        
        # input()
        # print("time for computing spectrum for one video: ", (time.time() - time_start_all))
        U_Matrices.append(U_curr[0])
        # print(U_Matrices)
    # ============================================ GraphRQI stage =======================================================
    data = U_Matrices[0]
    import joblib
    # mlp = joblib.load('GraphRQI_SKlearn_MLP')
    dim = 191 # MLP dimension
    mlp = joblib.load('myMLPmodel')
    data_shape = data.shape[0]
    pred_data = data
    if data_shape < dim:
        print("less ID in video clip")
        dif = dim - data_shape
        pred_data = np.pad(pred_data, [(0,dif), (0, dif)], mode='constant')
    elif data_shape > dim:
        print("more ID in video clip")
        pred_data = pred_data[:dim, :dim]
    res = mlp.predict(pred_data)
    # ============================================ predict stage ========================================================
    print(res)

