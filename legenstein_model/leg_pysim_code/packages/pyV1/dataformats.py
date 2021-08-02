'''
Dataformats for Stimulus and Response

    Malte Rasch    rasch@igi.tugraz.at
    Klaus Schuch   schuch@igi.tugraz.at
    
    July 2007
'''

import numpy
import tables
import utils.hdf5pickle as hdf5pickle
import copy
import learning.response2states as r2s
import os

class GenericParameterObject(object):
    """ Implements a generic Parameter class which is inherited by the
    AreaParameter classes. It has HDF5 saving capabilities """

    def __init__(self):
        self.save_attrs = [];     # list off attrs to be saved 
        self.attr_doc = {};


    def __str__(self):
        desc = "GenericParameterObject.\n"
        desc += " save attributes: \n" + self.save_attrs.__str__();
        return desc


    def save(self, filename, grpname=None, filters=tables.Filters(complevel=3, complib="zlib", shuffle=0, fletcher32=0), matlab=False):
        ''' saves save_attrs to HDF5 file '''

        if grpname is not None:
            if grpname[0]=='/':
                grpname = grpname[1:]
                if len(grpname)<1:
                    grpname = None


        if len(self.save_attrs):
            if grpname:
                h5file = tables.openFile(filename, mode = "a")

                try:
                    h5file.removeNode(h5file.root, grpname, recursive=True)
                except:
                    pass

                grp = h5file.createGroup(h5file.root, grpname)
            else:
                h5file = tables.openFile(filename, mode = "w")
                grp = h5file.root

                # make array for Tsim
                for attr in self.save_attrs:
                    h5file.createArray(grp, attr, self.__getattribute__(attr), self.attr_doc.get(attr,''));

            h5file.close()

        else:
            hdf5pickle.dumpDict(self.__dict__,filename,grpname,matlab=matlab)


    def saveMatlab(self, filename, grpname='', filters=tables.Filters(complevel=3, complib="zlib", shuffle=0, fletcher32=0)):
        """saves HDF5 files to be read in matlab"""
        self.save(filename, grpname=grpname, filters=filters,matlab=True)


    def load(self, filename, grpname=None):
        '''loads data from HDF5 file
        '''

        # THIS PRODUCES AN ERROR (why now?)
        #if grpname is not None:
        #    if grpname[0]=='/':
        #        grpname = grpname[1:]
        #        if len(grpname)<1:
        #            grpname = None

        if len(self.save_attrs):
            h5file = tables.openFile(filename, mode = "r");

            if grpname:
                grp = h5file.getNode(h5file.root, grpname)
            else:
                grp = h5file.root;


            # read the params
            for node in h5file.iterNodes(grp,'Array'):
                if node.name in self.save_attrs:
                    arr = numpy.array(node.read())
                    self.__setattr__(node.name, arr)

            h5file.close()

        else:
            self.__dict__ = hdf5pickle.loadDict(filename,grpname) 


# END GenericParameterObject 
#--------------------------------------------------



class TimeListDescription(tables.IsDescription):
    x     = tables.FloatCol()               # spiketimes or samples
    idx   = tables.UInt32Col()              # neuron indx 



class Channel(object):
    def __init__(self, data=numpy.empty(0), dt=-1, idx=-1):
        self.data = data
        self.dt = dt
        self.spiking = (dt<=0)
        self.idx = idx


    def __str__(self):
        strdata = ['%06.4f' % s for s in self.data]
        desc = '''  Channel
   data    : [%s]
   spiking : %s
   dt      : %s
        ''' % (strdata, self.spiking, self.dt)

        return desc


    class Description(tables.IsDescription):
        chanidx    = tables.UInt32Col();           # channel number NOTE THIS HAS TO BE THERE! ITS LIKE THE MATLAB INDEX
        spiking    = tables.Int8Col(1);            # whether spikign or not
        dt         = tables.FloatCol();            # time resolution
#        pos        = tables.FloatCol(shape=(1,3)); # x,y,z position
        idx        = tables.Int32Col(1);           # type of neuron somehow




class GenericDataObject(object):
    """ Implements a generic data object which is inherited by the
    Stimulus/Response objects. It has HDF5 saving capabilities """

    def __init__(self, channel_class=Channel):
        self.channel = [];
        self.channel_class = channel_class;
        self.save_attrs = [];     # list off attrs to be saved as array additionally to "channel"
        #self.attr_doc = {};
        self.Tsim = 0;

    def __str__(self):
        desc = '''  GenericDataObject
   channel        : [1x%s struct]
   Tsim          : %s
        ''' % (len(self.channel), self.Tsim)

        return desc


    def plot(self, channels=None, startT=0.0, endT=None, color='b',ioff=0):
        '''  plots the Data
        channels ... a list of channel indices (if None all channels are plotted
        '''
        if endT is None:
            endT=self.Tsim

        if channels!=None:
            ch = [self.channel[j] for j in channels]
            plot_channels(ch, [startT, endT],color=color,ioff=ioff)
        else:
            plot_channels(self.channel, [startT, endT],color=color,ioff=ioff)


    def reorganize(self):
        '''
        '''
        def cmpChanIdx(x, y):
            if x.idx == -1:
                if y.idx == -1:
                    return 0
                else:
                    return +1
            elif y.idx == -1:
                return -1
            else:
                return cmp(x.idx, y.idx)

        self.channel.sort(cmpChanIdx)


    def append(self, stim):
        '''  append another GenericDataObject
        stim ... the Object to append
        '''
        for c in stim.channel:
            self.appendChannel(c)

        if stim.Tsim > self.Tsim:
            self.Tsim = stim.Tsim


    def appendChannel(self, appchannel=None):
        '''  appends a channel 
        appchannel ... the channel to add (if None a empty channel will be added)#
        '''

        if appchannel is None:
            self.channel.append(self.channel_class());

        elif type(appchannel)==list:
            for ch in appchannel:
                self.channel.append(ch);
        else:
            self.channel.append(appchannel);


    def __len__(self):
        return len(self.channel);


    def __iter__(self):
        return self.channel


    def save(self, filename, grpname=None, filters=tables.Filters(complevel=3, complib="zlib", shuffle=0, fletcher32=0), channels=None):
        ''' saves data to HDF5 file '''

        if grpname is None:
            grpname = self.__class__.__name__
        else:
            if grpname[0]=='/':
                grpname = grpname[1:]
                if len(grpname)<1:
                    grpname = None


        if grpname:
            h5file = tables.openFile(filename, mode = "a")

            try:
                h5file.removeNode(h5file.root, grpname, recursive=True)
            except:
                pass

            grp = h5file.createGroup(h5file.root, grpname)
        else:
            h5file = tables.openFile(filename, mode = "w")
            grp = h5file.root

        if channels is None:
            channels = range(len(self.channel))
        assert(isinstance(channels,list))

        #save attributes with dumpDict first (otherwise it gets confused with array-attributes)
        dic = {}
        dic['save_attrs'] = {}
        for attr in self.save_attrs:
            dic['save_attrs'][attr] = self.__dict__[attr]

        hdf5pickle.dumpDict(dic,h5file, grp) #matlab?


        #save Tsim
        h5file.createArray(grp, 'Tsim', self.Tsim, "Duration of recording in sec")

        # create the tables
        chantab = h5file.createTable(grp, 'channel', self.channel_class.Description, 
                                     expectedrows=len(channels), filters=filters)

        names = chantab.description._v_names;
        avoidnames = [];

        if len(channels) > 0:
            expectedsizeinMB = sum([self.channel[c].data.nbytes for c in channels])/(2.0**20)

            dtype=tables.Atom.from_dtype(self.channel[channels[0]].data.dtype)
            chan_data = h5file.createVLArray(grp, 'ChanData', dtype, expectedsizeinMB=expectedsizeinMB)

        for chanIdx in channels:
            #fill channel table
            chan = self.channel[chanIdx]

            for j in range(len(names)):
                if j in avoidnames: 
                    continue

                if names[j] == 'chanidx':
                    chantab.row['chanidx'] = chanIdx
                else:
                    try:
                        chantab.row[names[j]] = eval('chan.' + names[j])
                    except:
                        avoidnames.append(j) 

            chantab.row.append()
            chan_data.append(chan.data)

        chantab.flush()
        h5file.close()



    def savemat(self,filename,name=None,appendmat=True):
        """ saves data in a matfile (as list and index)"""

        import scipy.io
        import os
        
        if name is None:
            name = self.__class__.__name__

        #convert to matlab style format
        N = 0

        for ch in self.channel:
            N += len(ch.data)

        data = numpy.zeros(N)
        chidx = numpy.zeros(N)
        spiking = numpy.zeros(len(self.channel))        
        dt = numpy.zeros(len(self.channel))        

        s = 0
        for i in range(len(self.channel)):
            chdata = self.channel[i].data
            L = len(chdata)
            data[s:s+L] = chdata[:]
            chidx[s:s+L] = i+1
            s = s+L
            spiking[i] = self.channel[i].spiking
            dt[i] = self.channel[i].dt


        if len(name):
            name = '_'+name
        
        dic = {'data'+name:data,'chanidx'+name:chidx,'Tsim'+name: numpy.double(self.Tsim),'spiking'+name:spiking,'dt'+name:dt}

        for attr in self.save_attrs:
            dic[attr + name] = numpy.array(self.__dict__[attr])

        if appendmat:
            # NOTE: there is a parameter in scipy.io.savemat ("appendmat=True") but this doesn't work! scipy BUG!
            # thus for now: do it by hand 
            try:
                dic2=scipy.io.loadmat(filename)
                dic2.update(dic) # old data will be overwritten
                dic = dic2
                del dic['__globals__']
            except:
                pass

        scipy.io.savemat(filename,dic,appendmat=True)
        

    def load(self, filename, grpname=None, idx_list=None):
        '''loads data from HDF5 file
           idx_list ... list of channel idx that should be read
        '''

        if grpname is None:
            grpname = '/' + self.__class__.__name__
        
        h5file = tables.openFile(filename, mode = "r");

        if grpname:
            grp = h5file.getNode(h5file.root, grpname)
        else:
            grp = h5file.root;

        #try to read save_attrs
        if grp._v_groups.has_key('save_attrs'):
            dic = hdf5pickle.loadDict(h5file,grp._v_groups['save_attrs'])
            for attr in dic.keys():
                self.__dict__[attr] = dic[attr]
                self.save_attrs.append(attr)
        else:
            for attr in self.save_attrs:
                self.__dict__[attr] = h5file.getNode(grp, attr).read()

        try:
            chan_data = h5file.getNode(grp, 'ChanData')
            chan_data = chan_data.read()
        except:
            chan_data = None

        try:
            self.Tsim = h5file.getNode(grp, 'Tsim').read()
        except:
            pass
        self.channel=[]

        # read the tables
        for node in h5file.iterNodes(grp, 'Table'):
            #read the channels
            if node.name == 'channel':
                if idx_list is None:
                    self.channel = [self.channel_class() for idx in range(node.nrows)]

                    for row in node.iterrows():
                        s = row['chanidx']

                        if chan_data is not None:
                            self.channel[s].data = chan_data[s]

                        for colname in node.colnames:
                            try:
                                self.channel[s].__setattr__(colname, row[colname])
                            except:
                                print "something went wrong: %s " % colname
                else:
                    self.channel = []

                    for row in node.iterrows():
                        idx = row['idx']

                        if idx in idx_list:
                            c=self.channel_class()
                            s = row['chanidx']

                            if chan_data is not None:
                                c.data = chan_data[s]

                            for colname in node.colnames:
                                try:
                                    c.__setattr__(colname, row[colname])
                                except:
                                    print "something went wrong: %s " % colname

                            self.channel.append(c)

        h5file.close()


    def scpsave(self,fname,grpname=None,location='',savepath='figipc57:~/../workspace/malte/saves/pysim'):
        """ saves object to savepath/location/fname in HDF5 group grpname"""
        import os

        self.save(fname,grpname)
        r = os.system('scp %s %s/%s' % (fname, savepath,location))
        if not r:
            os.remove(fname)



#END GenericDataObject
#--------------------------------------------------



class GenericDataObjectLst(GenericDataObject):
    """ multiple responses with local and global attributes. Will
    save/load as a single Response in one file and group"""

    def __init__(self, item_lst = None):
        super(GenericDataObjectLst, self).__init__(channel_class=Channel)
        self.n_lst = []
        self.Tsim_lst = []
        self.save_attrs = ['n_lst','Tsim_lst','local_attrs']

        self.local_attrs = []
        
        if item_lst != None:
            self.append(item_lst)

        self._itemClass = GenericDataObject
    
    def __str__(self):
        desc = '''  GenericDataObjectLst
        channel       : %s
        Tsim          : %s
        ''' % (str(self.n_lst), str(self.Tsim_lst))
        return desc


    def append(self,item):
        """ adds a item object."""

        if type(item)!=list:
            item_lst = [item]
        else:
            item_lst = item
            
        for r in item_lst:
            
            self.n_lst.append(len(r))
            self.Tsim_lst.append(r.Tsim)
            super(GenericDataObjectLst, self).append(r)

            d = {}
            for a in r.save_attrs:
                d[a] = r.__getattribute__(a)
                
            self.local_attrs.append(d)
            

    def __iter__(self):
        l = []
        for idx in range(len(self.n_lst)):
            l.append(self[idx])
        return l.__iter__()


    def tolist(self):

        l =  [];
        for idx in range(len(self.n_lst)):
            l.append(self[idx])
        return l


    def __setitem__(self,idx,item):
        """ sets an response object"""
        idx = int(idx)

        if idx>len(self.n_lst) or idx<0: 
            raise Exception('idx out of bounds')

        n = numpy.cumsum(numpy.array(self.n_lst))

        if idx==0:
            start = 0
        else:
            start = n[idx-1]
 
        del self.channel[start:n[idx]]

        self.channel[start:start] = item.channel
        
        self.n_lst[idx] = len(item)
        self.Tsim_lst[idx] = item.Tsim

        d = {}
        for a in item.save_attrs:
            d[a] = item.__getattribute__(a)
        
        self.local_attrs[idx] = d


    def addAttr(self,attr,value):
        """ adds an  global attribute """
        if not self.__dict__.has_key(attr):
            self.save_attrs += [attr]

        self.__dict__[attr] = value

        
    def __getitem__(self,idx):
        """ global attributes will overwrite local (if both existent)!"""
        idx = int(idx)
        
        if idx>len(self.n_lst) or idx<0: 
            raise Exception('idx out of bounds')

        item = self._itemClass()
        item.Tsim=self.Tsim_lst[idx]

        n = numpy.cumsum(numpy.array(self.n_lst))

        if idx==0:
            start = 0
        else:
            start = n[idx-1]
            
        item.channel = self.channel[start:n[idx]]

        #get attrs right
        global_attrs = []
        for a in self.save_attrs:
            if not a in ['n_lst', 'Tsim_lst','local_attrs']:
                global_attrs += [a]

        #update local
        for (key,val) in zip(self.local_attrs[idx].keys(),self.local_attrs[idx].values()):
            item.__setattr__(key,val)
            item.save_attrs += [key]

        #global overwrites local!
        for attr in global_attrs:
            item.__setattr__(attr,self.__getattribute__(attr)) 
            item.save_attrs += [attr]
            
        return item

    def plot(self,*args,**kwds):

        import pylab

        for item in self:
            pylab.figure()
            item.plot(*args,**kwds)

            

    def save(self,*args,**kwds):

        self.save_attrs = list(set(self.save_attrs + ['n_lst', 'Tsim_lst','local_attrs']))
        super(GenericDataObjectLst,self).save(*args,**kwds)

        
    def load(self,*args,**kwds):
        
        super(GenericDataObjectLst,self).load(*args,**kwds)
        self.save_attrs = list(set(self.save_attrs))

        if not ('n_lst' in self.save_attrs):
            #make Response a GenericDataObjectLst
            d = {}
            for a in self.save_attrs:
                d[a] = self.__dict__.pop(a)

            self.save_attrs = ['n_lst','Tsim_lst','local_attrs']
            self.local_attrs = [d]
            self.n_lst = [len(self.channel)]
            self.Tsim_lst = [self.Tsim]

        else:
            try:
                self.n_lst = self.n_lst.tolist()
                self.Tsim_lst = self.Tsim_lst.tolist()
            except:
                pass


    def savemat(self,*args,**kwds):

        #matlab cannot save this
        idx = 0
        save_attrs = copy.copy(self.save_attrs)
        add_save_attrs = []
        for d in self.local_attrs:
            for k in d.keys():
                f =  '__LAIDX%d__%s' % (idx,k)
                add_save_attrs += [f]
                self.__dict__[f] = d[k]
            idx += 1

        self.save_attrs += add_save_attrs
        self.save_attrs.remove('local_attrs')

        super(GenericDataObjectLst,self).savemat(*args,**kwds)

        for a in add_save_attrs:
            del self.__dict__[a]

        self.save_attrs = save_attrs
        
    def reorganize(self):
        pass


#--------------------------------------------------------------

class Stimulus(GenericDataObject):
    def __init__(self, Tsim=0.5):
        super(Stimulus, self).__init__(channel_class=Channel)
        self.Tsim = Tsim;
        self.save_attrs = [];     # list off attrs to be saved as array additionally to "channel"
        #self.attr_doc = {};

    def __str__(self):
        desc = '''  Stimulus
        channel       : [1x%s struct]
        Tsim          : %s
        ''' % (len(self.channel), self.Tsim)
        return desc

 


        
class Response(GenericDataObject):
    def __init__(self, Tsim=0.5):
        super(Response, self).__init__(channel_class=Channel)
        self.Tsim = Tsim;
        self.save_attrs = [];     # list off attrs to be saved as array additionally to "channel"
        #self.attr_doc = {};

    def __str__(self):
        desc = '''  Response
        channel       : [1x%s struct]
        Tsim          : %s
        ''' % (len(self.channel), self.Tsim)
        return desc


    def plot(self,*args,**kwds):

        if not self.__dict__.has_key('typeidx'):
            super(Response,self).plot(*args,**kwds)
        else:
            # getLayerIdx() was done
            import pylab

            ch = numpy.array(self.channel)
            
            for l in range(len(self.layers)):
                
                idx = numpy.where(self.layeridx==l)[0]

                channels = ch[idx]

                pylab.subplot(len(self.layers), 1, l+1)

                typeidx = self.typeidx[idx]
                
                exc = idx[numpy.where(typeidx==0)[0]]
                inh = idx[numpy.where(typeidx==1)[0]]
                s = 0
                ispks = numpy.r_[0:0.]
                ispks_i = numpy.r_[0:0.]
                espks = numpy.r_[0:0.]
                espks_i = numpy.r_[0:0.]
                
                for channel in channels.tolist():

                    if typeidx[s]:
                        ispks = numpy.r_[ispks, channel.data]
                        ispks_i = numpy.r_[ispks_i, s*numpy.ones(len(channel.data))]
                    else:
                        espks = numpy.r_[espks, channel.data]
                        espks_i = numpy.r_[espks_i, s*numpy.ones(len(channel.data))]
                    
                    s+=1

                pylab.plot(ispks,ispks_i,'r.',*args,**kwds)
                pylab.hold('on')
                pylab.plot(espks,espks_i,'k.',*args,**kwds)


            pylab.title("Layer "+ self.layers[l])


    def makeLayerMovie(self,fname='responseMovie',nsteps=50,fps=2):
        
        states =r2s.getLayerwiseStates(self,sampling=numpy.linspace(0,self.Tsim,nsteps))
        
        figh = r2s.plotLayerwiseStates(states)

        f_lst = []
        for (s,h) in enumerate(figh):
            curname = fname + '%02d' % s  
            curname = curname + '.png'
            h.savefig(curname)
            f_lst.append(curname)

        os.system("mencoder 'mf://%s*.png' -mf type=png:fps=%d -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg" % (fname,int(fps),fname))

        for f in f_lst:
            os.remove(f)


class ResponseLst(GenericDataObjectLst):
    def __init__(self, rsp_lst=None):
        super(ResponseLst, self).__init__(rsp_lst)

        self._itemClass = Response
        
    def __str__(self):
        desc = '''  ResponseLst
        channel       : %s
        Tsim          : %s
        ''' % (str(self.n_lst), str(self.Tsim_lst))
        return desc


class StimulusLst(GenericDataObjectLst):
    def __init__(self, stim_lst=None):
        super(StimulusLst, self).__init__(stim_lst)

        self._itemClass = Stimulus

    def __str__(self):
        desc = '''  StimulusLst
        channel       : %s
        Tsim          : %s
        ''' % (str(self.n_lst), str(self.Tsim_lst))
        return desc


#---------------------------------------------------------

def loadResponse(fname, grpname=None):
    resp = Response()
    resp.load(fname, grpname)
    return resp



def plot_channels(channels, xlim=None, scale=None, color='k',ioff=0):
    ''' plot the given channels 
    '''
    import pylab

    if scale is None:
        m = 0;
        for channel in channels:
            if not channel.spiking:
                m = numpy.max(numpy.r_[m, numpy.abs(channel.data)])
        scale = 1.5*m

    spks = numpy.r_[0:0.]
    spks_i = numpy.r_[0:0.]

    plotlist = [];
    i = ioff
    for channel in channels:
        if channel.spiking:
            spks = numpy.r_[spks, channel.data]
            spks_i = numpy.r_[spks_i, i*numpy.ones(len(channel.data))]
        else:
            tend = channel.dt*len(channel.data);
            plotlist.append(numpy.arange(0, tend, channel.dt))
            plotlist.append(channel.data/scale + i)
            plotlist.append(color)
        i+=1

    if len(plotlist):
        pylab.plot(*plotlist)

    if len(spks):
        pylab.plot(spks, spks_i, '.',color=color)

    if len(plotlist):
        if len(plotlist)/3 > 1:
            pylab.setp(pylab.gca(), ylim=[-0.5, len(channels)-0.5])
    else:
        pylab.setp(pylab.gca(), ylim=[-0.5, len(channels)-0.5])

    if xlim:
        pylab.setp(pylab.gca(), xlim=xlim)



if __name__ == '__main__':
    from datetime import datetime
    import pylab

    print "Testing Dataformats"
    
    Tsim = 50.0
    freq = 20.0
    response = Response(Tsim)

    dt = 1.0e-2
    d = numpy.sin(numpy.arange(0, Tsim, dt)*5)

    for i in range(100):
        response.channel.append(Channel(numpy.random.uniform(0, Tsim, freq*Tsim)))
        response.channel.append(Channel(d, dt))
        response.channel.append(Channel())

    pylab.figure()
    response.plot()

    aresponse = Response(Tsim)

    for i in range(10):
        aresponse.channel.append(Channel(d**2, dt))
#        resp.channel[i].data = s

    pylab.figure()
    aresponse.plot()

    start = datetime.today()
    response.save(filename='test.h5')
    dt = datetime.today() - start
    print "Speichern (neu):", dt


    start = datetime.today()
    response1 = Response();
    response1.load(filename='test.h5')
    dt = datetime.today() - start
    print "Laden (neu):", dt


    #---------
    print "Testing Dataformats.ResponseLst"

    response.glu = 1
    response.save_attrs += ['glu']

    aresponse.blub = 1
    aresponse.save_attrs += ['blub']

    rspLst = ResponseLst([aresponse,response])
    pylab.figure()
    rspLst[0].plot()

    rspLst.plot()

    rspLst[0]= aresponse

    rspb = rspLst[1]

    print rspb.save_attrs
    pylab.figure()
    rspb.plot()
    
    rspLst.addAttr('bla',1)

    print rspLst.save_attrs
    print rspLst.local_attrs
    
    rspLst.save(filename='test_lst.h5')
    print rspLst

    rspLst2 = ResponseLst()
    rspLst2.load(filename='test_lst.h5')

    rspLst2.plot()

    print rspLst2
    print rspLst2.local_attrs

    rspLst.savemat('test.mat')

    fname= '/home/workspace/malte/saves/pysim/sim_WscaleAndLRW_seed1434103_getStandardMovieStm_17x17/WscaleAndLRW_seed1434103_getStandardMovieStm_17x17.298'
    rsp = Response()
    rsp.load(fname,'/Response')
    rsp.plot()
