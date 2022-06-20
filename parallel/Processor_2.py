import cv2

def run(sys_args, dict_frame):
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
    global g_frame
    cap = cv2.VideoCapture(sys_args.video_path)
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
            dict_frame[frame_id] = frame
            frame_id += 1