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
        # frame: image frame.
        ret_val, frame = cap.read()
        # start working when have image.
        if ret_val:
            futures = None
            if sys_args.resize:
                frame = cv2.resize(frame,(sys_args.size))
            # bounding box and ID infomation
            # ========== run MOT module ==========
            res = System.MOT_run(frame, frame_id, format=sys_args.format_str)

            if sys_args.show:
                System.show(frame, res)

            # ========== run TP module ==========
            System.update_traj(res)
            if sys_args.future:
                futures = System.get_future_traj()

            # ========== run BC module ==========
            System.BC_run(futures)

            # ========== run OT module ==========
            if System.OT_run(frame):
                break
            frame_id += 1
        else:
            break
