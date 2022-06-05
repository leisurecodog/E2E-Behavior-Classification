from os import system
import cv2
import system_parser
import system_class


if __name__ == '__main__':
    sys_args = system_parser.get_parser()
    System = system_class.DrivingBehaviorSystem()
    cap = cv2.VideoCapture(sys_args.video_path if sys_args.demo == "video" else sys_args.camid)
    frame_id = 0

    while True:
        # ret_val: True -> Read image, False -> No image
        # frame: just frame
        ret_val, frame = cap.read()
        if ret_val:
            futures = []
            if sys_args.resize:
                frame = cv2.resize(frame,(sys_args.size))

            # bounding box and ID infomation
            res = System.MOT_run(frame, frame_id, format=sys_args.format_str)
            if sys_args.show:
                System.show(frame, res)
            System.update_traj(res)
            if sys_args.future:
                futures = System.get_future_traj()
            System.BC_run(futures)

            if System.OT_run(frame):
                break
            frame_id += 1
        else:
            break
