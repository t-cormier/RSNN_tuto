import pyplusplus

NAME    = 'pypcsim_special'
HEADERS = [ ]
DEFINES = [ "MPICH_IGNORE_CXX_SEEK" ,"BOOST_HAS_THREADS" ,"BOOST_PYTHON_MAX_ARITY=30" ]
CFLAGS  = [ '-pthread' ]
DEPEND_MODULES = []

def specify( M, options ):

    M.class_( 'BioFeedRewardGenAlpha' ).include()
    M.class_( 'BioFeedRewardGen' ).include()
    M.class_( 'BioFeedRewardGenDblExp' ).include()
    M.class_( 'ReadoutRewardGen' ).include()
    M.class_( 'RewardGeneratorAdder' ).include()
    M.class_( 'RewardGenerator2' ).include()
    M.class_( 'RewardGenerator' ).include()
    
    return M
    
############################################################################################################

def postprocessing( M, V, options ):
    pass
