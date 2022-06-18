from xml.etree.ElementTree import TreeBuilder
from TP_module.util import prGreen
import time
# 
def run(frame_dict, objdet_dict, MOT_dict):
    from MOT_module.MOT import MOT
    module = MOT()
    # ==================================================
    while True:
        if module.frame_id in frame_dict:
            data = frame_dict[module.frame_id]
            module.run(data, objdet_dict, MOT_dict)
