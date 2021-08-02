import tables
import numpy
import pdb

def put_string(h5,where,name,string,ext_name):
    ''' put a sting in a hdf5 file'''

    atom = tables.StringAtom(itemsize=len(string))
    ca = h5.createCArray(where,name,atom,(1,),ext_name)
    ca[:] = string[:]


def array_to_atom(array):
    ''' converts a numpy.array into a pytables atom'''

    return tables.Atom.from_dtype(array.dtype)



def put_array(h5,where,name,ar,ext_name):
    ''' put a numpy.array in a hdf5 file'''

    atom = array_to_atom(ar)

    ca = h5.createCArray(where,name,atom,ar.shape,ext_name)
    ca[:] = ar[:]


def put_python(h5,where,name,seq,ext_name):
    ''' put a variable in a hdf5 file'''

    # We have CArray support so try to store as a CArray
    # use array to check homogeneity of sequence
    # will raise if not homogeneous
    tmp = numpy.array(seq)

    # assure numerical type
    if tmp.dtype in (numpy.object,'c'):
        raise TypeError, "non numerical python sequence for %s" % str(seq)

    atom = array_to_atom(tmp) 

    ca = h5.createCArray(where,name,atom,tmp.shape,ext_name)
    ca[:] = tmp[:]



def dumpDict(d, h5, where=None, filters=None, matlab=False):
    """write a dictionary, d, to an open pytables file object, h5, at Group object or path string, where"""
    
    if where is None or where=='':
        (h5, grp, bClose) = openFile(h5, 'w', where, filters)
    else:
        (h5, grp, bClose) = openFile(h5, 'a', where, filters)

    import pickle as p

    keys = d.keys()
    names = []
    ext_names = []
    non_hdf_names = []

    # handle non-hdf5 names
    for key in keys:

        if matlab and type(key)==tuple:
            key = reduce(lambda x,y: x + '__' +y, key);

        
        if not type(key)==str or not key[0].isalpha()  or (len(key)>=12 and key[0:12]=='non_hdf_name'):
            # pickle key
            names+= ['non_hdf_name%d' % len(non_hdf_names)]
            non_hdf_names+=[names[-1]]
            ext_names+=['#'+p.dumps(key)]
        else:
            # key is ok as hdf5 node name
            names+=[key]
            ext_names+=['#']

    for i in range(len(keys)):
        key = keys[i]
        name = names[i]
        ext_name = ext_names[i]

        if type(d[key])==type(numpy.array([])):
            put_array(h5,grp,name,d[key],ext_name)

        elif (matlab and (numpy.isscalar(d[key]) and not type(d[key])==str)):
            h5.createArray(grp,name,numpy.array(d[key]).astype(numpy.float64), ext_name )

        elif (matlab) and d[key]==None:
            #what to do with none? just 0 
            h5.createArray(grp,name,numpy.array(0).astype(numpy.float64), ext_name )

        elif (not matlab) and (d[key] in (int,float)):
            h5.createArray(grp,name,numpy.array(d[key]), ext_name )

#        elif type(d[key]) in (str,):
#            #h5.createArray(grp,name,d[key], ext_name )
#            put_string(h5,grp,name,d[key],ext_name)
        elif type(d[key])==dict:
            # recurse with group named key
            g = h5.createGroup(grp,name, title=ext_name)
            dumpDict(d[key],h5,g,filters,matlab=matlab)
        else:
            try:
                put_python(h5,grp,name,d[key],ext_name)
            except Exception, x:
                if matlab:
                    # recurse with group named key (for matlab) CAUTION: cannot be loaded in python
                    if type(d[key]) in (list,tuple):
                        g = h5.createGroup(grp,name, title=ext_name)
                        ii = 0
                        for dk in d[key]:
                            if type(dk)<>dict:
                                n = name + '____' + str(ii)
                                tmpdk = {n : dk}
                                dumpDict(tmpdk,h5,g,filters,matlab=True)
                            else:
                                g2 = h5.createGroup(g,name + '____' + str(ii), title=ext_name+ '__' + str(ii))
                                dumpDict(dk,h5,g2,filters,matlab=True)
                            ii+=1
                                                        
                    elif type(d[key]) in (str,):
                        put_string(h5,grp,name,d[key],ext_name)
                    else:
                        g = h5.createGroup(grp,name, title='OBJECT: ' + ext_name)
                        try:
                            dumpDict(d[key].__dict__,h5,g,filters,matlab=True)
                        except:
                            pdb.set_trace()
                else:
                    #print "put_python Exception: %s" % str(x)
                    #print "Pickling unsupported hdf5 type at key '%s' title '%s' for storage." % (name,ext_name)
                    put_string(h5,grp,name,p.dumps(d[key]),ext_name+p.dumps('Pickled'))
                
    if bClose:
        h5.close()


def createGroupStructure(h5, grp, where=None):

    if where is None or where=='':
        where = h5.root
    elif type(where)==str:
        if where[0] != '/':
            where = '/' + where
        where = h5.getNode(where)

    if grp.find('/')==0:
       grp = grp[1:]

    node=where
    clist = grp.split('/')

    for c in clist:
        try:
            node = h5.createGroup(node, c)
        except:
            node = h5.getNode(node, c)

    return node


def openFile(h5, mode, where=None, filters=None):
    bClose=False

    if type(h5)==str:
        if filters==None:
            filters = tables.Filters(complevel=5,complib='zlib')

#        h5 = tables.openFile(h5, mode, filters=filters)
        h5 = tables.openFile(h5, mode)
        bClose = True

    if where is None or where=='':
        g = h5.root
    elif type(where)==str:
        if mode != 'r':
            try:
                h5.removeNode(h5.root, where, recursive=True)
            except:
                pass

            g = createGroupStructure(h5, where)
        else:
            g = h5.getNode(where)
    else:
         g = where

    return (h5, g, bClose)


def loadDict(h5, where='', filters=None):
    """ where is a path or group """

    (h5, grp, bClose) = openFile(h5, 'r', where, filters)

    import pickle as p

    d = {}

    def unmangleKey(name,title):
        """ unmangle key work done by DictToHDF5 to support all hashables in hdf name,title"""
        from StringIO import StringIO
        import pickle as p

        if title=='#':
            return (name,False)

        s = StringIO(title[1:])

        if name[0:12]=='non_hdf_name':
            putkey = p.load(s)
        else:
            putkey = name

        #was it a pickled object also?
        pickled = False
        try:
            pickstr = p.load(s)
        except EOFError:
            pass
        else:
            if pickstr=='Pickled':
                pickled = True
            else:
                print "Warning: Garbage after non_hdf_name title"

        return (putkey,pickled)


    # process leaves
    for key,leaf in grp._v_leaves.iteritems():
        putkey,pickled = unmangleKey(key,leaf.title)
        if pickled:
            d[putkey] = p.loads(leaf.read()[0])
        else:
            d[putkey] = leaf.read()

            if type(d[putkey]) == type(numpy.array(1)):
                if d[putkey].shape == ():
                    d[putkey]=d[putkey].tolist()

    # process subgroups:
    for key,group in grp._v_groups.iteritems():
        putkey,pickled = unmangleKey(key,group._v_title)
        if pickled:
            print "Warning: ignoring group returned pickled==True"

        # recurse down the group
        d[putkey] = loadDict(h5,group)

    if bClose:
        h5.close()

    return d


if __name__ == '__main__':
    testdict = {'hello':numpy.array([1,2,3.0]),
                'subdict':{'test1':1.0,
                           't2':'string',
                           'tuple':(1.0,2.0,4.0),
                           'subsubdict':{'test2':numpy.array([0,0,0,1]),
                                         3.4:8.9}},
                3.4:{'$%^&':{'subx':1.0},
                     'x':1.0},
                '$$%#$':(1.0,3.4),
                1.0:[1.0,{3.4:1.0},[1,2,3],'cdf'],
                2.0:set([1,2,3])
               }

    testdict1 = {'hello':numpy.array([1,2,3.0]),
                 'subdict':{'test1':1.0,
                            't2':'string',
                            'tuple':(1.0,2.0,4.0),
                            'subsubdict':{'test2':numpy.array([0,0,0,1]),
                                          3.4:8.9}},
                 3.4:{'aaa':{'subx':1.0},
                      'x':1.0},
                 'bbb':(1.0,3.4),
                 1.0:[1.0,{3.4:1.0},[1,2,3],'cdf'],
                 2.0:set([1,2,3])
                }

