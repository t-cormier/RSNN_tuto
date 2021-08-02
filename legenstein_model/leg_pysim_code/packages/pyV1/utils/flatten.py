from __future__ import generators

def canLoopOver(maybeIterable):
    try:
        iter(maybeIterable)
    except:
        return 0
    else:
        return 1
    
def isScalar(item):
    return not canLoopOver(item)

def flatten(sequence, scalarp=isScalar, result=None):
    if result is None: result = []
    for item in sequence:
        if scalarp(item):
            result.append(item)
        else: 
            flatten(item, scalarp, result)
    return result
    
    
def flattenGenerator(sequence, scalarp=isScalar):
    for item in sequence:
        if scalarp(item):
            yield item
        else:
            for subitem in flatten(item, scalarp):
                yield subitem

if __name__=='__main__':
    t = [[1,2,3],[3,4,5]]
    print flatten(t)

