from OT_module.OT import OT
import time
def p4(dict_frame, dict_objdet, dict_OT):
    module_OT = OT()
    frame_id = 0
    while True:
        if frame_id in dict_objdet:
            frame = dict_frame[frame_id]
            t1 = time.time()
            module_OT.run(frame, dict_objdet[frame_id])
            # dict_OT[frame_id] = module_OT.OTS.msg
            # g_frame = frame.copy()
            # UI_window.setup_control(frame)
            dict_OT.update({frame_id:module_OT.OTS.msg})
            frame_id += 1