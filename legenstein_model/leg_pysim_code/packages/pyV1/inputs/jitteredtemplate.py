from inputgenerator import *
import numpy

from datetime import datetime

class JitteredTemplateStimulus(Stimulus):
    def __init__(self, Tsim=0.5, actualTemplate=[1]):
        super(JitteredTemplateStimulus, self).__init__(Tsim)
        self.actualTemplate = actualTemplate

    def __str__(self):
        desc = '''  JitteredTemplateStimulus
  channel        : [1x%s struct]
  Tsim          : %s
  actualTemplate : %s\n''' % (len(self.channel), self.Tsim, self.actualTemplate)
        return desc


class JitteredTemplate_Segment(object):
    def __init__(self):
        self.template = []

    def __str__(self):
        desc = 'template : [1x%s struct]' % (len(self.template))
        return desc


class JitteredTemplate_Template(object):
    def __init__(self):
        self.st = []

    def __str__(self):
        desc = 'st : %s' % (self.st)
        return desc


class JitteredTemplate(InputGenerator):
    '''  jittered spike templates '''

    def __init__(self, **kwds):
        '''  nChannels  ... number of spike trains (channels)
  Tstim      ... length of spike trains
  jitter     ... jitter to add to each spike
  nTemplates ... number of templates per segment [1 x #Seg]
  nSpikes    ... number of spikes per template (uniformly dist) [1 x #Seg]
  freq       ... frequency of poisson spike train templates [1 x #Seg]
  segment    ... stores the actual spike templates
    '''
        self.nChannels = kwds.get('nChannels', 1)
        self.Tstim = float(kwds.get('Tstim', 1.0))
        self.jitter = float(kwds.get('jitter', 4e-3))
        self.nTemplates = kwds.get('nTemplates', [2])
        self.nSpikes = kwds.get('nSpikes', [])
        self.freq = [float(f) for f in kwds.get('freq', [20])]
        self.segment = []            # stores the actual spike templates

        self.generateTemplate()


    def __str__(self):
        desc = '''  JITTERED_TEMPLATE
  nChannels  : %s
  Tstim      : %s
  jitter     : %s
  nTemplates : %s
  nSpikes    : %s
  freq       : %s
  segment    : [1x%s struct] 
''' % (self.nChannels, self.Tstim, self.jitter, self.nTemplates, self.nSpikes, self.freq, len(self.segment))
        return desc


    def generateTemplate(self):
        self.nSegments = len(self.nTemplates)

        for s in range(self.nSegments):
            self.segment.append(JitteredTemplate_Segment())

            for i in range(self.nTemplates[s]):
                self.generateSegment(self.freq[s], s, i)


    def generateSegment(self, freq, s, i):
        tau_refract=3e-3
        
        lambd = 1/freq-tau_refract

        self.segment[s].template.append(JitteredTemplate_Template())
        self.segment[s].template[i].st=[]

        for j in range(self.nChannels):
            st=[];
            if len(self.nSpikes)>0:
                while len(st)==0:
                    st=numpy.cumsum(tau_refract + numpy.random.exponential(lambd, (1,self.nSpikes[s])))
                    st=st.compress((st<(self.Tstim/self.nSegments)).flat)
                    
            elif len(self.freq)>0:
                m  = numpy.ceil(5*freq*self.Tstim)
                st = numpy.cumsum(tau_refract + numpy.random.exponential(lambd, (1, m)))
                st = st.compress((st<=(self.Tstim/self.nSegments)).flat)

                st = st + s*self.Tstim/self.nSegments;
                self.segment[s].template[i].st.append(st.tolist())


    def generateChannel(self, freq, s, i):
        pass
    
    
    def generate(self, ti = None):
        '''generates a JitteredTemplateStimulus object'''
        
        nSegments = len(self.nTemplates)
        if not ti:
            ti = [int( numpy.random.uniform(0,self.nTemplates[i],1)) for i in range(nSegments)]
        stimulus = JitteredTemplateStimulus(self.Tstim, ti);
        
        for i in range(self.nChannels):
            stimulus.channel.append(Channel())
            
            st = []
            for s in range(nSegments):
                st += self.segment[s].template[ti[s]].st[i]
            st = numpy.asarray(st)

            if self.jitter > 0.0:
                st = st + numpy.random.randn(len(st))*self.jitter
                st = st[st>=0.0]
                st = st[st<=self.Tstim]
                st.sort()
                
            stimulus.channel[i].data = st
            
        return stimulus


    def plot(self, stimulus=None, COLOR=1, FS=1):
        '''Plotting the Template and a generated input'''
        import pylab

        if stimulus is None:
            stimulus=self.generate()
            
        pylab.figure()
        
        DEFCOLS = [];
        if COLOR != 0:
            DEFCOLS = [(1,0,0),(0,1,0),(0,0,1),(1,0.5,0),(0.3,0,0.8),(0.6,0,0),(0,0,0.6),(0,0.5,0),(0,1,1),(1,1,0),(1,0,1),(0,0,0)]
            
        col=0
        maxnt=0
        nSegments = len(self.segment) 
       
        for s in range(nSegments):
            maxnt=max(maxnt, len(self.segment[s].template))
            for t in range(len(self.segment[s].template)):
                col=col+1
                if COLOR == 1:
                    if col < len(DEFCOLS):
                        self.segment[s].template[t].col = DEFCOLS[col]
                    else: 
                        self.segment[s].template[t].col = (0,0,0)
                elif COLOR == 2:
                    if t < len(DEFCOLS):
                        self.segment[s].template[t].col = DEFCOLS[t]
                    else: 
                        self.segment[s].template[t].col = (0,0,0)
                        
                    
    
        sp1=pylab.subplot(211)
        MAXY=maxnt*self.nChannels
        L=self.Tstim/nSegments        
        for s in range(nSegments):
            for t in range(maxnt):
                if t < len(self.segment[s].template):
                    for j in range(len(self.segment[s].template[t].st)):
                        for spike in self.segment[s].template[t].st[j]:
                            pylab.plot(numpy.array([spike, spike]), MAXY-1-(numpy.array([[-0.3],[0.3]])+t*self.nChannels+j), color=self.segment[s].template[t].col, linewidth=2)
                else:
                    pylab.text((s+0.5)*L, MAXY-(t*self.nChannels+1), 'only ' +str(len(self.segment[s].template)) + ' templates', horizontalalignment='center', verticalalignment='center', fontsize=10*FS)                                
                
        for t in range(maxnt):
            pylab.plot([0,self.Tstim], MAXY-(numpy.array([0.5,0.5])+t*self.nChannels), color='k', linewidth=1)
        
        for s in range(nSegments-1):
            if COLOR > 1: 
                c = (0,0,0)
            else: 
                c = (0.5,0.5,0.5)        
            pylab.plot(numpy.array([1,1])*L*(s+1), MAXY-numpy.array([0.3, maxnt*self.nChannels+0.5]),color=c,linewidth=1,linestyle='--')
    
        for s in range(nSegments):
            pylab.text(L*(s+0.5), MAXY-0.2, str(s+1)+'. segment', verticalalignment='bottom', horizontalalignment='center', fontsize=10*FS)
    
        pylab.axis('tight')
        pylab.setp(pylab.gca(), xlim=[0,self.Tstim], ylim=[-1,maxnt*self.nChannels+0.5], xticks=[], yticks=[])   
        pylab.title('possible spike train segments',fontweight='bold',fontsize=10*FS)
    
        sp2=pylab.subplot(212)
        pylab.hold(1)
        MAXY=self.nChannels
            
        for j in range(self.nChannels):
            ST=stimulus.channel[j].data[:]
            for s in range(nSegments):
                t1=s*L
                t2=t1+L;
                st = [si for si in ST if (si>t1 and si<=t2)]            
                for spike in st:
                    pylab.plot([spike, spike], MAXY-1-(numpy.array([[-0.3],[0.3]])+j), color=self.segment[s].template[stimulus.actualTemplate[s]].col, linewidth=2)
        
        if nSegments > 1:
            for s in range(nSegments-1):
                if COLOR > 1: 
                    color = (0, 0, 0)
                else: 
                    c = (0.5, 0.5, 0.5)        
                pylab.plot(numpy.array([1,1])*L*(s+1), MAXY-1-(numpy.array([0.3, self.nChannels+0.5])), color=c, linewidth=1, linestyle='--')

        for s in range(nSegments):
            pylab.text(L*(s+0.5), MAXY-0.3, 'template '+str(stimulus.actualTemplate[s]), verticalalignment='bottom', horizontalalignment='center', fontsize=10*FS)
        
        pylab.axis('tight')
        pylab.setp(pylab.gca(), xlim=[0,self.Tstim], ylim=[-1,self.nChannels+0.5], yticks=[])
        pylab.xlabel('time [sec]', fontsize=10*FS)
    
        if self.nChannels > 1: 
            tit='resulting input spike trains'
        else: 
            tit='resulting input spike train'
            
        pylab.title(tit, fontweight='bold',fontsize=10*FS)
    
        pylab.setp(sp1,position=[0.07, 0.4, 0.86, 0.5])
        pylab.setp(sp2,position=[0.07, 0.1, 0.86, 0.2])



class JitteredTemplateDFT(JitteredTemplate):
    '''jittered spike templates with different frequencies of templates'''

    def __init__(self, **kwds):
        '''  nChannels  ... number of spike trains (channels)
  Tstim      ... length of spike trains
  jitter     ... jitter to add to each spike
  nTemplates ... number of templates per segment [1 x #Seg]
  nSpikes    ... number of spikes per template (uniformly dist) [1 x #Seg]
  freq       ... frequency of poisson spike train templates [1 x max(nTemplates)]
  segment    ... stores the actual spike templates
    ''' 
        self.nChannels = kwds.get('nChannels', 1)
        self.Tstim = float(kwds.get('Tstim', 1.0))
        self.jitter = float(kwds.get('jitter', 4e-3))
        self.nTemplates = kwds.get('nTemplates', [2])
        self.nSpikes = kwds.get('nSpikes', [])
        self.freq = [float(f) for f in kwds.get('freq', [20, 20])]
        self.segment = []            # stores the actual spike templates
        self.generateTemplate()


    def __str__(self):
        desc = '''  JITTERED_TEMPLATE DFT (different frequences of templates)
  nChannels  : %s
  Tstim      : %s
  jitter     : %s
  nTemplates : %s
  nSpikes    : %s
  freq       : %s
  segment    : [1x%s struct] 
''' % (self.nChannels, self.Tstim, self.jitter, self.nTemplates, self.nSpikes, self.freq, len(self.segment))
        return desc


    def generateTemplate(self):
        self.nSegments = len(self.nTemplates)

        for s in range(self.nSegments):
            self.segment.append(JitteredTemplate_Segment())
            
            for i in range(self.nTemplates[s]):
                self.generateSegment(self.freq[i], s, i)



class JitteredTemplateDFCH(JitteredTemplate):
    '''jittered spike templates with different frequencies of channels'''

    def __init__(self, **kwds):
        '''  nChannels  ... number of spike trains (channels)
  Tstim      ... length of spike trains
  jitter     ... jitter to add to each spike
  nTemplates ... number of templates per segment [1 x #Seg]
  nSpikes    ... number of spikes per template (uniformly dist) [1 x #Seg]
  freq       ... frequency of poisson spike train templates [1 x arbitary number]
  segment    ... stores the actual spike templates
    ''' 
        self.nChannels = kwds.get('nChannels', 1)
        self.Tstim = float(kwds.get('Tstim', 1.0))
        self.jitter = float(kwds.get('jitter', 4e-3))
        self.nTemplates = kwds.get('nTemplates', [2])
        self.nSpikes = kwds.get('nSpikes', [])
        self.freq = [float(f) for f in kwds.get('freq', [20, 20])]
        self.segment = []            # stores the actual spike templates
        self.generateTemplate()


    def __str__(self):
        desc = '''  JITTERED_TEMPLATE DFCH (different frequences of channels)
  nChannels  : %s
  Tstim      : %s
  jitter     : %s
  nTemplates : %s
  nSpikes    : %s
  freq       : %s
  segment    : [1x%s struct] 
''' % (self.nChannels, self.Tstim, self.jitter, self.nTemplates, self.nSpikes, self.freq, len(self.segment))
        return desc

    def generateTemplate(self):
        self.nSegments = len(self.nTemplates)

        for s in range(self.nSegments):
            self.segment.append(JitteredTemplate_Segment())
            
            for i in range(self.nTemplates[s]):
                self.generateSegment(self.freq[i], s, i)
# TODO


class JitteredTemplateRndInit(JitteredTemplate):
    '''jittered spike templates with different frequencies of templates
the initial input in the first Tinit seconds will be a random Poisson spike train of rate finit'''
    pass


class JitteredTemplateDetection(JitteredTemplate):
    '''jittered spike templates with '''
    def generate(self):
        '''generates a JitteredTemplateStimulus object'''        
        nSegments = len(self.nTemplates)
        ti = [int( numpy.random.uniform(0,self.nTemplates[i],1)) for i in range(nSegments)]
        stimulus = JitteredTemplateStimulus(self.Tstim, ti);
        
        for i in range(self.nChannels):
            stimulus.channel.append(Channel())
            st = []
            for s in range(nSegments):
                if ti[s] == 0:
                    tau_refract=3e-3
                    lambd = 1./self.freq[s]-tau_refract
                    m  = numpy.ceil(5*self.freq[s]*self.Tstim)
                    
                    spikes = numpy.cumsum(tau_refract + numpy.random.exponential(lambd, (1, m)))
                    spikes = spikes.compress((spikes <= (self.Tstim/self.nSegments)).flat)
                    spikes = spikes + s*self.Tstim/self.nSegments;
                    st += spikes.tolist()
                else:
                    st += self.segment[s].template[ti[s]].st[i]
                
            st = numpy.asarray(st)
            if self.jitter > 0.0:
                st = st + numpy.random.randn(len(st))*self.jitter
                st = st[st>=0.0]
                st = st[st<=self.Tstim]
                st.sort()
            stimulus.channel[i].data = st
            
        return stimulus
    

if __name__ == '__main__':
#    import pylab
#    jtemp=JitteredTemplate(Tstim=1.0, nChannels=3, nTemplates=[2,2,3,2], jitter=4e-3, freq=[10,20,10,30])
#    jtemp1=JitteredTemplate(Tstim=1.0, nChannels=1, nTemplates=[2,2,2], jitter=4e-3, freq=[20, 20, 20])
#    jtemp2=JitteredTemplateDFT(Tstim=1.0, nChannels=1, nTemplates=[2,3,3], jitter=4e-3, freq=[10, 30, 10])
#    jtemp2=JitteredTemplateDFCH(Tstim=1.0, nChannels=1, nTemplates=[2,3], jitter=4e-3, freq=[10, 30, 10, 40])
#    stim1=jtemp1.generate()
#    stim2=jtemp2.generate()

#    t_start=datetime.today()
#    r=[jtemp2.generate() for i in range(0, 1000)]
#    print 'duration:', datetime.today()-t_start

    jtempdetection1=JitteredTemplateDetection(Tstim=1.0, nChannels=1, nTemplates=[2,2,2,2,2,2], jitter=1e-3, freq=[20, 20, 20, 20, 20, 20])
    s1=jtempdetection1.generate()
    
    jtempdetection1.plot(s1)
    
#    jtemp1.plot(stim1)
#    jtemp1.plot(stim1, COLOR=2)
#    jtemp2.plot(stim2)
#    jtemp2.plot(stim2, COLOR=2)
#    pylab.savefig('jitTemp')
#    pylab.show()
    
        
    
    
