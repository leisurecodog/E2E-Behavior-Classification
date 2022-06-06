import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # ==================== System argument ====================
    parser.add_argument('--video_path', default='/media/rvl/D/Work/fengan/Dataset/CEO/front/201126152525.MOV', type=str)
    parser.add_argument('--demo', default='video', type=str)
    parser.add_argument('--camid', default=0, type=int)
    parser.add_argument('--dashcam', default=False, type=bool)
    parser.add_argument('--show', default=False, type=bool)
    # ==================== MOT argument ====================
    parser.add_argument('--traj_model_path', default='./TP_module/weights', type=str)
    parser.add_argument('--future', default=False, type=bool)
    parser.add_argument('--format_str', default='bbox', type=str)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--size', default=(1080, 720), type=tuple)
    return parser.parse_args()
