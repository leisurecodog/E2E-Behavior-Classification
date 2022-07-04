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

def writer(d,lock):
    # print(queue)
    for i in range(100):
        # queue.put(i)
        # lock.acquire()
        d[i] = i
        # lock.release()
    # print "worker", queue.qsize()
def reader(d,lock):
    for i in range(100):
        # print(queue.get_top())
        # lock.acquire()
        print(i in d)
        # lock.release()
    
m = Manager()
# q = m.MyQueue()  # This is process-safe
ll = m.list()
lock = m.Lock()
ll.append(0)
d = m.dict()
wp = Process(target = writer, args = (d,lock,))
rp = Process(target = reader, args = (d,lock,))
rp.start()
wp.start()

wp.join()
rp.join()
print(ll[0])