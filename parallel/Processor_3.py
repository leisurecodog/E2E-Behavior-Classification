import cv2
import system_parser
def run(*params):
    video_path, dict_frame, lock, signal = params
    '''
    function of Input_reader:
        Description:
            Read frame from video and put each frame into dict_frame container.
        Input:
            sys_args: All args using for system is there, more detail arguments please see system_parser.py
            dict_frame: A dictionary from torch.mp.manager.dict, it responses for storing frame.
        Output:
            None
    '''
    sys_args = system_parser.get_parser()
    # cap = cv2.VideoCapture(sys_args.video_path)
    # print(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        '''
        ret_val: { True -> Read image, False -> No image }
        frame: image frame.
        '''
        ret_val, frame = cap.read()
        if ret_val:
            g_frame = frame # TODO
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            dict_frame.update({frame_id:frame})
            frame_id += 1
        