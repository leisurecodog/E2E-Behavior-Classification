from multiprocessing import Manager
import time
def normal_dict(nums):
    n_dict = dict()
    t1 = time.time()
    # write test
    for i in range(nums):
        n_dict[i] = i
    t_write = time.time() - t1

    # read test
    t1 = time.time()
    for i in range(nums):
        data = n_dict[i]
    t_read = time.time() - t1

    return t_read, t_write

def manager_dict(nums):
    m = Manager()
    md = m.dict()
    t1 = time.time()
    # write test
    for i in range(nums):
        md[i] = i
    t_write = time.time() - t1

    # read test
    t1 = time.time()
    for i in range(nums):
        data = md[i]
    t_read = time.time() - t1

    return t_read, t_write

if __name__ == '__main__':
    nums = 1000
    nd_r, nd_w = normal_dict(nums)
    print("Normal Dictionary in Python, Read: {0:.10f} \t Write: {0:.10f}".format(nd_r, nd_w))
    md_r, md_w = manager_dict(nums)
    print("Manager Dictionary in Python, Read: {:.10f} \t Write: {:.10f}".format(md_r, md_w))
    
