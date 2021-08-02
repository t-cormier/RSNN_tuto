#
# Point PCSIM_SOURCE_DIR to the directory pcsim root directory; 
# that is where setup.py and MANIFEST.in are located.
#
SET( PCSIM_SOURCE_DIR "$ENV{HOME}/pcsim" )

#
# The name of this PCSIM extension module (which is the output of 'make all') 
# will be called ${PCSIM_MODULE_NAME}
#
SET( PCSIM_MODULE_NAME "reward_gen" )

#
# List here all "home" directories of other pcsim extension modules
# that this PCSIM extension module depends on 
# (location of CMakeLists.txt and python_interface_specification.py). 
# Leave it blank if there are not any dependencies. 
#
SET( MODULE_DEPENDENCIES "")

#
# Set the path to the lib<module>.so libraries for each PCSIM extension module 
# that this module depends on. This should be the location of the installed 
# shared object (OR dll file under windows).   
#
SET( MODULE_DEPENDENCIES_LIBRARYPATH "")

#
# Please list all sources here:
#
SET( MODULE_SOURCES
  src/BioFeedRewardGenAlpha.cpp
  src/BioFeedRewardGen.cpp
  src/BioFeedRewardGenDblExp.cpp
  src/ReadoutRewardGen.cpp
  src/RewardGeneratorAdder.cpp
  src/RewardGenerator2.cpp
  src/RewardGenerator.cpp
)
