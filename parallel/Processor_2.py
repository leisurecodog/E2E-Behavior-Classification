
def run(*params):
    import cv2
    dict_frame, dict_objdet, dict_OT, lock, signal = params
    from OT_module.OT import OT
    # import time
    module_OT = OT()
    frame_id = 0
    while True:
        if frame_id in dict_objdet:
            # 讀取影像與bounding boxes
            bbox = dict_objdet[frame_id]
            frame = dict_frame[frame_id]
            
            if bbox is not None:    
                # ======================== OT Work ======================
                module_OT.run(frame, bbox)
                dict_OT.update({frame_id:module_OT.OTS.msg})
            frame_id += 1