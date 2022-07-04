def run(*params):
    dict_frame, dict_objdet, dict_OT, lock = params
    from OT_module.OT import OT
    import time
    module_OT = OT()
    frame_id = 0
    entry_time = 0
    total_fps = 0
    counter = 0
    while True:
        if frame_id in dict_objdet:
            frame = dict_frame[frame_id]
            t1 = time.time()
            module_OT.run(frame, dict_objdet[frame_id])
            # lock.acquire()
            dict_OT.update({frame_id:module_OT.OTS.msg})
            # lock.release()
            print("OT done \t {}".format(frame_id))
            t2 = time.time() - t1

            frame_id += 1