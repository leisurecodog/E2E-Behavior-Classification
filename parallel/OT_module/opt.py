from argparse import ArgumentParser

def set_opt():
    opt = ArgumentParser()
    opt.add_argument('--yolact_edge', action='store_false')
    opt.add_argument('--obj_det', action='store_false')
    opt.add_argument('--save_video', action='store_false')
    opt.add_argument('--video_path', type=str, default='201126152425.MOV')
    return opt.parse_args()