
def run(*params):
    import cv2
    dict_frame, dict_objdet, dict_OT, lock, signal = params
    from OT_module.OT import OT
    import time
    module_OT = OT()
    frame_id = 0
    entry_time = 0
    total_fps = 0
    counter = 0
    while True:
        if frame_id in dict_objdet:
            bbox = dict_objdet[frame_id]
            frame = dict_frame[frame_id]
            if bbox is None:
                continue
            t1 = time.time()
            module_OT.run(frame, bbox, True)
            # cv2.imshow('ttttt', frame)
            # cv2.waitKey(1)
            # lock.acquire()
            dict_OT.update({frame_id:module_OT.OTS.msg})
            # lock.release()
            # print("OT done \t {}".format(frame_id))
            t2 = time.time() - t1
            frame_id += 1
            # if signal.value == frame_id:
            #     print("ot module average time:", module_OT.exe_time / module_OT.counter)
            #     return 