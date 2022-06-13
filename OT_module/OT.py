from TP_module.util import prGreen
import time
import numpy as np

class OT:    
    def __init__(self):
        msg = "Initializing OT Module..."
        prGreen(msg)
        from OT_module.main import inference, set_opt
        self.inference_ptr = inference
        self.OT_args = set_opt()
        self.counter = 0
        self.exe_time = 0

    def run(self, objdet, frame):
        st = time.time()
        frame = self.inference_ptr(objdet, frame)
        self.counter += 1
        self.exe_time += (time.time() - st)
        # return frame