import argparse
import os, sys
import os.path as osp
import time
import cv2
import torch

from loguru import logger

# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
# from yolox.utils.visualize import plot_tracking
# from yolox.tracking_utils.timer import Timer
from MOT_module.tracker.byte_tracker import BYTETracker


# =========================================yolov5========================
from numpy import random
from MOT_module.models.experimental import attempt_load
from MOT_module.utils.general import check_img_size, plot_one_box,set_logging
from MOT_module.utils.torch_utils import select_device
# from yolo_detect import detect
from MOT_module import yolo_detect

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
t_detector = 0
t_matcher = 0

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/1.MOV", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    # parser.add_argument(
    #     "-f",
    #     "--exp_file",
    #     default=None,
    #     type=str,
    #     help="pls input your expriment description file",
    # )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def evaluate_bdd100k(predictor, bdd100k_folder, args):
    global names, colors, imgsz
    tracker = BYTETracker(args, frame_rate=args.fps)
    files = os.listdir(bdd100k_folder)
    txt_path = bdd100k_folder[:-4] + 'preds'
    files = [f for f in files if f.find('.txt') == -1]
    # print(len(files))
    for dir_name in sorted(files):
        # print(dir_name)
        loc = os.path.join(bdd100k_folder, dir_name)
        tmp = os.path.join(txt_path, dir_name)
        txt_file = os.path.join(tmp + ".txt")
        print(txt_file)
        frame_id = 1
        results = []
        for img_name in sorted(os.listdir(loc)):

            img_loc = os.path.join(loc, img_name)
            print(img_loc)
            frame = cv2.imread(img_loc)
            # cv2.imshow("t", frame)
            # cv2.waitKey(0)
            img_info = {}
            # print(predictor is None, imgsz is None, names is None, frame is None)
            # print(frame.shape)
            outputs = yolo_detect.detect(predictor, imgsz, names, frame)
            img_info['height'], img_info['width'] = frame.shape[:2]
            img_info['raw_img'] = frame
            
            if outputs is not None:
                
                online_targets = tracker.update(reversed(outputs), [img_info['height'], img_info['width']], [img_info['height'], 
                img_info['width']])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )        
            frame_id += 1
        with open(txt_file, 'w') as f:
            for strs in results:
                f.write(strs)
                # print(strs)
                # input()           

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []
    
    for frame_id, img_path in enumerate(files, 1):
        
        outputs, img_info = predictor.inference(img_path, timer)
        
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
           
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            # online_im = plot_tracking(
            #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            # )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, 1.0/(timer.average_time / frame_id))))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def imageflow_demo(predictor, vis_folder, current_time, args):
    global names, colors, imgsz, avg_time, t_detector, t_matcher

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1]).replace("MOV","mp4")
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 1
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        
        if ret_val:
            # frame = cv2.resize(frame,(1080, 720))
            # outputs, img_info = predictor.inference(frame, timer)
            timer.tic()
            img_info = {}
            outputs = yolo_detect.detect(predictor, imgsz, names, frame)
            # t_detector += infer_time
            img_info['height'], img_info['width'] = frame.shape[:2]
            img_info['raw_img'] = frame

            if outputs is not None:
                t2 = time.time()
                online_targets = tracker.update(reversed(outputs), [img_info['height'], img_info['width']], [img_info['height'], 
                img_info['width']])
                t_matcher += time.time() - t2                
               
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # print(tlwh, tid)
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    #     online_tlwhs.append(tlwh)
                    #     online_ids.append(tid)
                    #     online_scores.append(t.score)
                    #     results.append(
                    #         f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    #     )
                timer.toc()
                
                # online_im = plot_tracking(
                #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1.0/timer.average_time)    
            else:
                timer.toc()
                online_im = img_info['raw_img']
            
            if args.save_result:
                vid_writer.write(online_im)

            
            # for *xyxy, conf, cls in reversed(outputs):
            #     # label = '%s %.2f' % (names[int(cls)], conf)
            #     a = int(xyxy[0])
            #     b = int(xyxy[1])
            #     c = int(xyxy[2])
            #     d = int(xyxy[3])                
            #     plot_one_box(xyxy, online_im, label=str(conf),color=(0,255,0), line_thickness=2)
            #     cv2.putText(online_im, str(conf.cpu().numpy()), (a, b), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
            
            # cv2.imshow('tracking res', online_im)
            # ch = cv2.waitKey(0)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    # print("custom average FPS: ", 1.0/(avg_time/frame_id))
    print("timer average time(ms): ", timer.average_time)
    print("model inferenced time: ", (t_detector*1000)/(frame_id), "ms")
    print("matching time: ", (t_matcher*1000)/(frame_id), "ms")

def load_yolov5(rt=True):
    global names, colors, imgsz
    weights = 'MOT_module/weights/s_best.pt'
    set_logging()
    device = select_device('')

    # load yolov5 model
    model_v5 = attempt_load(weights, map_location=device)
    # get image size for feed of yolov5
    imgsz = check_img_size(640, s=model_v5.stride.max())
    
    # check cpu utility
    half = device.type != 'cpu'
    if half:
        model_v5.half()

    names = model_v5.module.names if hasattr(model_v5, 'module') else model_v5.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    if rt:
        return model_v5, imgsz, names
    else:
        return model_v5

def main(args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    # output_dir = osp.join(exp.output_dir, args.experiment_name)
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = 'YOLOv5_outputs'
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    # if args.trt:
    #     args.device = "gpu"
    # args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    # logger.info("Args: {}".format(args))

    # if args.conf is not None:
    #     exp.test_conf = args.conf
    # if args.nms is not None:
    #     exp.nmsthre = args.nms
    # if args.tsize is not None:
    #     exp.test_size = (args.tsize, args.tsize)

    # model = exp.get_model().to(args.device)
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    # model.eval()

    # if not args.trt:
    #     if args.ckpt is None:
    #         ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    #     else:
    #         ckpt_file = args.ckpt
    #     logger.info("loading checkpoint")
    #     ckpt = torch.load(ckpt_file, map_location="cpu")
    #     # load the model state dict
    #     model.load_state_dict(ckpt["model"])
    #     logger.info("loaded checkpoint done.")

    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)

    # if args.fp16:
    #     model = model.half()  # to FP16

    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = osp.join(output_dir, "model_trt.pth")
    #     assert osp.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    #     decoder = None
# ====================================================================================================
    predictor = load_yolov5()
# ====================================================================================================
    current_time = time.localtime()
    if args.demo == "image":
        # image_demo(predictor, vis_folder, current_time, args)
        evaluate_bdd100k(predictor, "/home/fengan/Desktop/Dataset/BDD100K MOT/images20-track-val-1/bdd100k/images/track/val/", args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)

# def caller(video_path):
#     args = make_parser().parse_args()
#     print(video_path)
#     args.path = video_path
#     args.save_result = True
#     main(args)

# if __name__ == "__main__":
#     args = make_parser().parse_args()
#     # exp = get_exp(args.exp_file, args.name)
#     main(args)
