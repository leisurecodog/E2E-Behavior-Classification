import torch
import numpy as np
import time



from MOT_module.utils.datasets import LoadStreams, LoadImages,letterbox
from MOT_module.utils.general import non_max_suppression, scale_coords
from MOT_module.utils.torch_utils import select_device
# import cv2

def detect(model, imgsz, names, source):
    # source = cv2.resize(source, (512, 256))
    # set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # print(type(source))
    im0s = source
    # print("sorce:", source.shape)
    # print(imgsz)
    img = letterbox(source, new_shape=imgsz)[0]
    # print("img:",img.shape)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print(img.shape, type(img))
    pred = []
    timers = 0
    t1 = time.time()
    with torch.no_grad():
        # ttt1 = time.time()
        pred = model(img, augment=False)[0]
        timers = time.time() - t1
        # print("time.time(): ", time.time()-ttt1)
    # print(pred.cpu().shape)
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    for i, det in enumerate(pred):  # detections per image
        im0 = im0s
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # print(img.shape[2:])
            # print(im0.shape)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            return det

