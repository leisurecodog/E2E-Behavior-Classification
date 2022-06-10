
def worker(model, name, *params):
    if name == "pinet":
        x, y = model(params)
        print(x, y)
        
        
    elif name == 'yolo':
        a, b, c, d = params
        net_pred = model(a,b,c,d)
        print(net_pred)
        
        

if __name__ == '__main__':
    import torch.multiprocessing as mp
    manager = mp.Manager()
    
    mp.set_start_method('spawn', force=True)
    
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

    lane_agent = Agent()
    lane_agent.cuda()
    lane_agent.evaluate_mode()
    lane_agent.load_weights(895, "tensor(0.5546)")

    frame = np.zeros((720, 1080, 3))
    
    # frame_tensor = torch.from_numpy(frame).cuda().float()
    # batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
    # moving_statistics = {"conf_hist": []}
    # extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,"moving_statistics": moving_statistics}
    # outputs = yolo_detect.detect(object_predictor, imgsz, names, frame)
    with torch.no_grad():
        p1 = mp.Process(target=worker, args=(lane_agent, 'pinet', frame,))
        p2 = mp.Process(target=worker, args=(yolo_detect.detect, 'yolo', object_predictor, imgsz, names, frame))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
