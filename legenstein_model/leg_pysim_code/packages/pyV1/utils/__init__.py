__version__ = '0.1'

__all__ = ['flatten']

from flatten import flatten as flatten
from flatten import canLoopOver as canLoopOver
from flatten import isScalar as isScalar
 
from datetime import datetime
from numpy import mean

def sub_time(t1, t2):
    return (t1 - t2).seconds+1e-6*(t1-t2).microseconds;


def timedTest(function):
    
    def wrapper(*args, **kw):
        all = []
        count = 5
        
        for i in range(count):
            t0 = datetime.today()        

            res = function(*args, **kw)

            t1 = datetime.today()
            all.append(sub_time( t1, t0 ))
        
        print min(all), '(min),', max(all), '(max),', mean(all), '(mean) sec CPU time'
        
        return res
    
    return wrapper
