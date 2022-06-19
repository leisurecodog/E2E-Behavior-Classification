import os

def ID_check(obj, name=''):
    s = "Your obj name={}, ID is {}.".format(name, id(obj))
    print(s)