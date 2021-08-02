#!/bin/env python
#
# Build and setup script for PCSIM extension modules
#
#  march 2008  Dejan Pecevski, Thomas Natschlaeger
#
import os, sys, shutil, getopt

def usage():
    print "Usage <python> pcsim_extension.py -h --help --module=<module_name> -m <module_name> --debug -g \n--outdir=<output directory> -o <output directory>  <create_template | build | clean | wipe | mrproper | install | <other targets> >"

def help():
    pass

def SYSTEM_ASSERT( execstr, verb = True ):
    if verb:
        print '>>>> cd',os.getcwd()
        print '>>>>',execstr
    retval = os.system( execstr )
    if not retval == 0:
        sys.stderr.write( 'BUILD PROCESS FAILED. ERROR = %s\n' % ( retval ) )
        sys.stderr.write( 'Command: %s\n' % ( execstr ) )
        sys.exit(-1)

 
def create_extension_template(module_name, pcsim_home, outdir):
    #
    # Needs to have PCSIM_HOME environment variable set
    #
    if os.path.exists( os.path.join( ".", module_name ) ):
        raise Exception('Cannot create pcsim extension module template at %s. The directory already exists.')
    shutil.copytree(os.path.join( pcsim_home, 'extension_template' ),
                    os.path.join( outdir, module_name ) )


def build(build_type, *targets):
    if not os.path.exists('build'): 
        os.mkdir('build')
    curr_dir = os.getcwd()
    os.chdir('build')
    SYSTEM_ASSERT('cmake -D CMAKE_BUILD_TYPE:STRING=' + build_type + ' ..')
    SYSTEM_ASSERT('make generate_code')
    SYSTEM_ASSERT('make rebuild_cache')
    SYSTEM_ASSERT('make ' + ' '.join(targets))
    os.chdir(curr_dir)
    

def clean(*args):
    shutil.rmtree(os.path.join('build'))
    pass


if __name__ == '__main__':
    #
    # Process the options
    #
    
    optlist, args = getopt.getopt(sys.argv[1:] , 'M:P:Hhgo',
        ['module=', 'pcsim_home=', 'debug' , 'outdir='] )

    PCSIM_HOME = '/usr'
    CMAKE_BUILD_TYPE = 'Release'
    module = 'my_pcsim_module'
    outdir = os.getcwd()
    
    if os.environ.has_key("PCSIM_HOME"):
        PCSIM_HOME = os.environ["PCSIM_HOME"]
    elif os.environ.has_key("PCSIM_ROOT_DIR"):
        PCSIM_HOME = os.environ["PCSIM_ROOT_DIR"]        

    
    for opt, arg in optlist:
        if opt in ('-H' '-h' '--help'):
            help()
            sys.exit(0)    
        elif opt in ( '-P' '--pcsim_home'):    
            PCSIM_HOME = arg
        elif opt in ('-M' '--module'):
            module=arg
        elif opt in ( '-g' '--debug'):
            CMAKE_BUILD_TYPE = 'Debug'
        elif opt in ( '-o' '--outdir'):
            outdir = arg
    
    if len(args) == 0:
        usage()
        sys.exit(
                 )
    cmd = args[0]

    
    if cmd == 'create_template':
        create_extension_template(module, PCSIM_HOME, outdir)
    elif cmd in ( 'wipe' 'mrproper' ):
        clean(*args[1:])
    else:
        build(CMAKE_BUILD_TYPE, *args[1:])
    