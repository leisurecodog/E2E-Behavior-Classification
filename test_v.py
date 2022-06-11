import cv2
import time
def worker(func, name, *params):
    # print(name)
    nums = 100
    if name == 'pinet':
        nums = 10000
    for i in range(nums):
        st = time.time()
        func(*params)   
        st2 = time.time()
        print(name, st2-st)
    # if name == 'pinet':
    #     a, b = params
    #     for i in range(10):
    #         st = time.time()
    #         func(a, b)
    #         st2 = time.time()
    #         print("worker", st2-st)
    # if name == 'yolo':
    #     a, b, c, d = params
    #     st = time.time()
    #     func(a,b,c,d)
    #     st2 = time.time()
    #     print("worker", st2-st)
    
    # print(result)
    # if name == 'yolo':
    #     _, _, _, frame = params
    #     result.cpu().detach().numpy()
    #     for i in range(len(result)):
    #         x1, y1 = result[i][:2]
    #         x2, y2 = result[i][2:4]
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0))
    #     cv2.imshow("777777", frame)
    #     cv2.waitKey(0)
        
        

if __name__ == '__main__':
    import torch.multiprocessing as mp
    manager = mp.Manager()
    
    mp.set_start_method('spawn', force=True)
    # from OT_module.yolact_edge_project.eval import load_yolact_edge, prep_display
    # from OT_module.yolact_edge_project.utils.augmentations import FastBaseTransform

    from MOT_module.tools.demo_track_yolov5 import load_yolov5
    object_predictor, imgsz, names = load_yolov5(rt=True)
    

    import numpy as np
    
    from MOT_module import yolo_detect
    img_info = {}
    results = {}
    
    from OT_module.PInet.test import PInet_test
    from OT_module.PInet.agent import Agent
    import torch
    
    # yolact_model = load_yolact_edge()
    # yolact_model.detect.use_fast_nms = True
    # lane_agent = Agent()
    # lane_agent.cuda()
    # lane_agent.evaluate_mode()
    # lane_agent.load_weights(895, "tensor(0.5546)")

    
    frame = cv2.imread('/home/rvl/Pictures/modified_graphRQI_Result.png')
    
    # frame_tensor = torch.from_numpy(frame).cuda().float()
    # batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
    # moving_statistics = {"conf_hist": []}
    # extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,"moving_statistics": moving_statistics}
    
    # st = time.time()
    # PInet_test(lane_agent, frame)
    # input()
    # st2 = time.time()
    # yolo_detect.detect(object_predictor, imgsz, names, frame)
    # st3 = time.time()

    with torch.no_grad():
        p1 = mp.Process(target=worker, args=(PInet_test, 'pinet', lane_agent, frame,))
        p2 = mp.Process(target=worker, args=(yolo_detect.detect, 'yolo', object_predictor, imgsz, names, frame))

        # p1 = mp.Process(target=PInet_test, args=(lane_agent, frame,))
        # p2 = mp.Process(target=yolo_detect.detect, args=(object_predictor, imgsz, names, frame))
        # p3 = mp.Process(target=worker, args=(yolact_model, 'yolact', batch, extras))
        sp = time.time()
        p1.start()
        p2.start()
        # p3.start()
        spp = time.time()
        p1.join()
        p2.join()
        
        # p3.join()
    print(st2-st, st3-st2)
    print(spp-sp)
