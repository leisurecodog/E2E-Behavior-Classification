from queue import Queue
import dis
import time
from multiprocessing import Process
from multiprocessing.managers import SyncManager

class MyQueue(Queue):
    def __init__(self):
        super().__init__()
    def get_top(self):
        if not self.empty():
            return self.queue[0]
        else:
            return -1

class MyManager(SyncManager):
    pass
MyManager.register("MyQueue", MyQueue)  # Register a shared Queue

def Manager():
    m = MyManager()
    m.start()
    return m

def writer(queue):
    # print(queue)
    for i in range(100):
        queue.put(i)
    # print "worker", queue.qsize()
def reader(queue):
    for i in range(100):
        print(queue.get_top())
    
m = Manager()
q = m.MyQueue()  # This is process-safe
wp = Process(target = writer, args = (q,))
rp = Process(target = reader, args = (q,))
wp.start()
rp.start()