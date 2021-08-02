import ipython1.kernel.api as kernel
from ipython1.config import cutils
import os
import time
from pyV1.utils.logger import logger


def restart(clusterno=0):
    # short cut
    configFile = None
    if clusterno:
        configFile = 'clusterconf%d.py' % clusterno
        ipdir = cutils.getIpythonDir()
        configFile = os.path.join(ipdir,configFile)

    cl = Cluster(ClusterConfig(configFile=configFile))
    try:
        cl.connect()
        cl.stop(dt=1.0, waitafter=8.0)
    except:
        pass
    cl.start(dt=1.0)
    return cl


def stop(clusterno=0):
    # short cut
    configFile = None
    if clusterno:
        configFile = 'clusterconf%d.py' % clusterno
        ipdir = cutils.getIpythonDir()
        configFile = os.path.join(ipdir,configFile)

    cl = Cluster(ClusterConfig(configFile=configFile))
    cl.connect()
    cl.stop(dt=1.0, waitafter=8.0)
    return cl


def start(clusterno=0):
    # short cut
    configFile = None
    if clusterno:
        configFile = 'clusterconf%d.py' % clusterno
        ipdir = cutils.getIpythonDir()
        configFile = os.path.join(ipdir,configFile)

    cl = Cluster(ClusterConfig(configFile=configFile))

    if cl.connect():
        cl.start(dt=1.0)
        cl.connect()

    return cl


def run(cmd,NS_lst,resultNames=[],cmd_all='',blocked=True,clusterno=0):
    """ run a task on default cluster"""
    
    cl = start(clusterno)
    
    rc = cl.getRemoteControllerClient()
    rc.resetAll()

    if len(cmd_all):
        rc.executeAll(cmd_all)

    #create tasks
    task_lst = [];

    for s in range(len(NS_lst)):
        task_lst.append(kernel.Task(cmd, resultNames=resultNames, setupNS = NS_lst[s]))

    tc = cl.getTaskControllerClient()
    t_start = time.time()
    tids = [tc.run(tk) for tk in task_lst]

    logger.info('Submitted %d tasks to cluster %d' % (len(tids),clusterno))

    if blocked:
        tc.barrier(tids)

        cl.checkError(tids)

        res = []
        for tid in tids:
            res.append(tc.getTaskResult(tid))

        t_end = time.time() 
        logger.info('done, duration: %1.1f sec' % (t_end - t_start))

        return res
    else:
        return (cl, tids)



class ClusterConfig(dict):
    def __init__(self, configFile=None, controller=None, engines=None, numEngines=0, sshx=None):
        if configFile is None:
            ipdir = cutils.getIpythonDir()
            configFile = os.path.join(ipdir,'clusterconf.py')

        execfile(configFile,self)
        self.pop('__builtins__','')

        if controller is not None:
            self['controller']=controller

        if sshx is not None:
            self['sshx']=sshx

        self['ncluster'] = self.get('ncluster',False)

        if engines is not None:
            if isinstance(engines, dict):
                self['engines']=engines
            elif isinstance(engines, list):
                self['engines']=dict()
                for e in engines:
                    self['engines'][e]=numEngines
        elif numEngines > 0:
            for e in self['engines'].keys():
                self['engines'][e]=numEngines


    def getControllerHost(self):
        return self['controller']['host']

    def getEnginePort(self):
        return self['controller']['engine_port']

    def getRemoteControllerPort(self):
        return self['controller']['rc_port']

    def getTaskControllerPort(self):
        return self['controller']['task_port']

    def getEngines(self):
        return self['engines']

    def getSSHX(self):
        return self.get('sshx',os.environ.get('IPYTHON_SSHX','sshx'))

    def getNcluster(self):
        return self['ncluster']


class Cluster(object):
    def __init__(self, clusterConfig, dt=0.5, use_mpd=False):
        self.dt = dt
        self.use_mpd = use_mpd
        self.max_wait_time=300

        # read configuration
        self.sshx = clusterConfig.getSSHX()
        self.contHost = clusterConfig.getControllerHost()
        self.engine_port = clusterConfig.getEnginePort()
        self.rc_port = clusterConfig.getRemoteControllerPort()
        self.task_port = clusterConfig.getTaskControllerPort()
        self.engines = clusterConfig.getEngines()
        self.ncluster = clusterConfig.getNcluster()

        # setup logfile
        ipdir = cutils.getIpythonDir()
        logdir_base = os.path.join(ipdir,'log')
        if not os.path.isdir(logdir_base):
            os.makedirs(logdir_base)
        logfile = os.path.join(logdir_base,'ipcluster')
        self.logfile = '%s-%s' % (logfile,os.getpid())
        self.__running = False

    def __startController(self, dt=None, verbose=True):
        if dt is None:
            dt=self.dt
        if verbose:
            print 'Starting controller:'
            print '  Starting controller on %s' % self.contHost
        contLog = '%s-con-%s-' % (self.logfile,self.contHost)

        if not self.ncluster:
            cmd = "ssh %s '%s' 'ipcontroller --engine-port=%s --remote-cont-port=%s --task-port=%s --logfile=%s' &" % \
                  (self.contHost,self.sshx,self.engine_port,self.rc_port,self.task_port,contLog)
            os.system(cmd)

        else:
            # on the ncluster
            import socket
            self.contHost = socket.gethostbyaddr(socket.gethostname())[2][0]
            self.contHost= os.popen('echo $HOSTNAME').read()[:-1]

            #self.task_port = 10113
            #self.rc_port = 10105

            cmd = "ipcontroller --engine-port=%s --remote-cont-port=%s --task-port=%s --logfile=%s &" % \
                  (self.engine_port, self.rc_port, self.task_port, contLog)

            print 'cmd:', cmd
            os.system(cmd)

        time.sleep(dt)


    def __startEngines(self, dt=None, verbose=True):
        if dt is None: dt=self.dt

        if not self.ncluster:
            if verbose:
                print 'Starting engines:   '
            self.nodecount=0

            for engineHost,numEngines in self.engines.iteritems():
                if verbose:
                    print '  Starting %d engine(s) on %s' % (numEngines,engineHost)
                engLog = '%s-eng-%s-' % (self.logfile,engineHost)

                self.nodecount+=numEngines
                for i in range(numEngines):
                    cmd = "ssh -x %s '%s' 'ipengine --controller-ip=%s --controller-port=%s --logfile=%s' &" % \
                              (engineHost,self.sshx,self.contHost,self.engine_port,engLog)
                    os.system(cmd)
        else:
            #ncluster
            engLog = '%s-eng' % (self.logfile)

            self.nodecount = int(os.popen('cat $PBS_NODEFILE|wc -l').read()[:-1])
            print 'nodecount:', self.nodecount
            if verbose:
                print 'Starting', self.nodecount, ' engines...'
            #cmd = "mpiexec ipengine --controller-ip=%s --controller-port=%s --mpi=mpi4py &" % \
            #           (self.contHost, self.engine_port)
            cmd = "mpiexec -nostdout ipengine --controller-ip=%s --controller-port=%s --mpi=mpi4py --logfile=%s &" % \
                       (self.contHost, self.engine_port, engLog)
            print 'cmd:', cmd
            os.system(cmd)

        time.sleep(2.0)
        self.rc = kernel.RemoteController((self.contHost, self.rc_port))
        wait_time=0
        while (len(self.rc.getIDs()) < self.nodecount and wait_time <= self.max_wait_time):
            print 'waiting...'
            time.sleep(5.0)
            wait_time+=5

        print 'started', len(self.rc.getIDs()), 'engines'

        if wait_time > self.max_wait_time:
            raise Exception('TIMEOUT: not all started!!!!!')

        time.sleep(dt)


    def __startMPDs(self, dt=None, verbose=True):
        if dt is None: dt=self.dt
        if verbose:
            print 'Starting mpds:   '
        for engineHost in self.engines.iterkeys():
            print '  Starting mpd on %s' % engineHost
            cmd = "ssh %s 'mpd' &" % (engineHost)
            os.system(cmd)
            time.sleep(dt)


    def start(self, startmpds=False, dt=None, waitafter=0.0, verbose=True):

        if self.isRunning():
            raise Exception("cluster already running")

        if dt is None: dt=self.dt

        self.__startController(verbose=verbose, dt=4.0)
        time.sleep(dt)

        if startmpds:
            self.__startMPDs(verbose=verbose)
            time.sleep(dt)

        try:
            self.__startEngines(verbose=verbose)
            time.sleep(dt)
        except Exception, inst:
            self.tc = kernel.TaskController((self.contHost,self.task_port))
            self.rc.activate()
            self.__running = True
            raise inst

        self.tc = kernel.TaskController((self.contHost,self.task_port))
        self.rc.activate()

        if waitafter>0.0:
            time.sleep(waitafter)

        print "Your cluster is up and running."
        self.__running = True


    def connect(self):
        try:
            self.rc = kernel.RemoteController((self.contHost,self.rc_port))
            self.tc = kernel.TaskController((self.contHost,self.task_port))

            # test if the cluster is really running
            self.rc.execute(0,'pass')

            self.__running = True
            return 0
        except:
            print "Your cluster is NOT running."
            return 1

    def getRemoteControllerClient(self):
        return self.rc

    def getTaskControllerClient(self):
        return self.tc

    def isRunning(self):
        return self.__running

    def stop(self, force=False, killmpds=False, dt=None, waitafter=0.0, verbose=True):
        if not self.__running:
            return

        if dt is None: dt=self.dt

        if verbose:
            print "Killing controller and engines..."
        if (not force) or self.ncluster:
            try:
                self.rc.killAll(controller=True)
                time.sleep(dt)
            except:
                force=True

        if force and (not self.ncluster):
            for engineHost in self.engines.iterkeys():
                if verbose:
                    print '  Killing engine on %s' % engineHost
                cmd = "ssh %s 'pkill -u `whoami` -f ipengine' " % (engineHost)
                os.system(cmd)
                time.sleep(dt)
            if verbose:
                print '  Killing controller on %s' % self.contHost
            cmd = "ssh %s 'pkill -u `whoami` -f ipcontroller' " % (self.contHost)
            os.system(cmd)
            time.sleep(dt)

        if killmpds:
            if verbose:
                print "Killing mpds..."
            for engineHost in self.engines.iterkeys():
                if verbose:
                    print '  Killing mpd on %s' % engineHost
                cmd = "ssh %s 'pkill -u `whoami` -f mpd' " % (engineHost)
                os.system(cmd)
                time.sleep(dt)

        if waitafter>0.0:
            time.sleep(waitafter)

        print "Your cluster has been successfully shut down."
        self.__running = False


    def checkError(self,tids):

        tc = self.getTaskControllerClient()
        for tid in tids:
            res = tc.getTaskResult(tid)
            if None<>res.failure:
                res.failure.printDetailedTraceback()
                res.failure.raiseException()
