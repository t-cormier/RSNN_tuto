

INCLUDE( module_recipe.cmake ) 

SET( CMAKE_VERBOSE_MAKEFILE OFF )


SET( SRCDIR     "src"     )
SET( CPPUNITDIR "cppunittests" )
SET( BUILDDIR   "build"   )


PROJECT( ${PCSIM_MODULE_NAME} )
SET( USER_PCSIM_SOURCE_DIR "${${PCSIM_MODULE_NAME}_SOURCE_DIR}" )

GET_FILENAME_COMPONENT( SRCDIR "${SRCDIR}" ABSOLUTE )
GET_FILENAME_COMPONENT( CPPUNITDIR "${CPPUNITDIR}" ABSOLUTE )

INCLUDE_DIRECTORIES( "." ${SRCDIR} ${CPPUNITDIR} ${BUILDDIR} )

INCLUDE( "${PCSIM_SOURCE_DIR}/config.cmake" )

SET( SIM_OBJ_HEADER )
FOREACH( cpp ${MODULE_SOURCES} )
	STRING( REPLACE ".cpp" ".h" tmp ${cpp} )
    SET( SIM_OBJ_HEADER ${SIM_OBJ_HEADER} ${tmp} )
ENDFOREACH( cpp )

#
# Figure out the module names of all modules that the module depends on
#
SET( DEP_MODULE_NAMES )
FOREACH( EXTENSION_HOME ${MODULE_DEPENDENCIES} )
	FILE( ${EXTENSION_HOME}/build/module_name.txt read DEP_MODULE_NAME )
	SET( DEP_MODULE_NAMES "${DEP_MODULE_NAMES} ${DEP_MODULE_NAME}" )
	INCLUDE_DIRECTORIES( "${EXTENSION_HOME}/src"
                         "${EXTENSION_HOME}/cppunittests"
                         "${EXTENSION_HOME}/build" )		 
ENDFOREACH( EXTENSION_HOME )

#
# Write down the file containing the name of the module 
#
FILE(WRITE build/module_name.txt "${PCSIM_MODULE_NAME}")

LINK_LIBRARIES( ${MODULE_DEPENDENCIES_LIBRARYPATH} )
 

#**************************************************************************************
#                Generating C++ files for registering SimObjects
#**************************************************************************************

SET( SIM_OBJ_REG_CPP      "${USER_PCSIM_SOURCE_DIR}/build/RegisterSimObjects.cpp" )
SET( SIM_OBJ_REG_INC_I    "${USER_PCSIM_SOURCE_DIR}/build/register_fields_includes.h" )
SET( SIM_OBJ_REG_FIELD_I  "${USER_PCSIM_SOURCE_DIR}/build/register_fields_calls.h" )
SET( SIM_OBJ_REGISTRATOR  "${USER_PCSIM_SOURCE_DIR}/build/${PCSIM_MODULE_NAME}SimObjectRegistrator.h" )

SET( DOXYFILE  "${PCSIM_SOURCE_DIR}/pcsim/doxyfile_for_registry_code" )
SET( GENSCRIPT "${PCSIM_SOURCE_DIR}/scripts/generate_simobject_registry_code.py" )

ADD_CUSTOM_COMMAND( 
  OUTPUT  ${SIM_OBJ_REG_CPP} ${SIM_OBJ_REG_INC_I} ${SIM_OBJ_REG_FIELD_I} ${SIM_OBJ_REGISTRATOR}
  COMMAND "${PYTHON_EXECUTABLE}" "${GENSCRIPT}"
      "--module=${PCSIM_MODULE_NAME}"  
      "--doxyfile=${DOXYFILE}"  
      "--doxygen=${DOXYGEN_EXECUTABLE}"  
      "--input=src"       
      "--include=${PCSIM_SOURCE_DIR}/pcsim/simcore ${PCSIM_SOURCE_DIR}/pcsim/construction ${PCSIM_SOURCE_DIR}/pcsim/simutils ${PCSIM_SOURCE_DIR}/pcsim/simobjects"
      "--pcsim_home=${PCSIM_SOURCE_DIR}" 
      --doxygenoutdir="${USER_PCSIM_SOURCE_DIR}/build" 
      --outdir="${USER_PCSIM_SOURCE_DIR}/build" 
      "--outfile=RegisterSimObjects.cpp"
  DEPENDS ${SIM_OBJ_HEADER}
  WORKING_DIRECTORY "${USER_PCSIM_SOURCE_DIR}"
)

INCLUDE_DIRECTORIES( "${PCSIM_SOURCE_DIR}/pcsim/simutils" 
                     "${PCSIM_SOURCE_DIR}/pcsim/simcore" 
                     "${PCSIM_SOURCE_DIR}/pcsim/construction" 
                     "${PCSIM_SOURCE_DIR}/pcsim/simobjects"
                     "${PCSIM_SOURCE_DIR}/cppunittests" )

INCLUDE_DIRECTORIES( build )

#****************************************************************************************************
#                                 generating Py++ wrapper code
#****************************************************************************************************

SET( MYPYPCSIM py${PCSIM_MODULE_NAME} )

FILE(GLOB SIMOBJECT_HEADERS "src/*.h" )

SET( PYPP_HEADER_FILE "${USER_PCSIM_SOURCE_DIR}/build/${MYPYPCSIM}.h" )

ADD_CUSTOM_COMMAND(
	OUTPUT  "${PYPP_HEADER_FILE}"
	COMMAND ${PYTHON_EXECUTABLE} "${PCSIM_SOURCE_DIR}/scripts/merge_includes.py" 
				--out="${PYPP_HEADER_FILE}" ${SIMOBJECT_HEADERS}
	DEPENDS ${SIMOBJECT_HEADERS}	
	WORKING_DIRECTORY ${USER_PCSIM_SOURCE_DIR}
)

SET( GEN_BP_WRAPPER_PY "${PCSIM_SOURCE_DIR}/scripts/generate_boost_python_wrapper_code.py" )

SET( MODPYPPMAIN "${USER_PCSIM_SOURCE_DIR}/build/${MYPYPCSIM}/${MYPYPCSIM}.main.cpp" )
SET( MODPYPPINCT "${USER_PCSIM_SOURCE_DIR}/build/${MYPYPCSIM}/${MYPYPCSIM}.indicator.cpp" )

ADD_CUSTOM_COMMAND(
	OUTPUT  ${MODPYPPMAIN} ${MODPYPPINCT}
	COMMAND ${PYTHON_EXECUTABLE} ${GEN_BP_WRAPPER_PY} 
	    --module=python_interface_specification	    
	    --name="${MYPYPCSIM}" 
	    --temp="${MYPYPCSIM}_temp"
	    --outdir=build
	    --pcsim_home=${PCSIM_SOURCE_DIR} 
	    -H"${PYPP_HEADER_FILE}" 
		-I"src" 
		-I"${PCSIM_SOURCE_DIR}/pcsim/simutils" 
		-I"${PCSIM_SOURCE_DIR}/pcsim/simcore" 
		-I"${PCSIM_SOURCE_DIR}/pcsim/construction" 
		-I"${PCSIM_SOURCE_DIR}/pcsim/simobjects" 
		-I"${BOOST_INCLUDE_PATH}" 
		-I"${MPI_INCLUDE_PATH}" 
		-I"${GSL_INCLUDE_PATH}" 
		-I"${PCSIM_SOURCE_DIR}/python"
	DEPENDS "${PYPP_HEADER_FILE}"	
	WORKING_DIRECTORY ${USER_PCSIM_SOURCE_DIR}
)

ADD_LIBRARY( generate_wrapper_code "${MODPYPPINCT}" )

ADD_CUSTOM_TARGET( generate_code )
ADD_DEPENDENCIES( generate_code generate_wrapper_code )


# ****************************************************************************************************
#                                   Building the C++ shared library
# ****************************************************************************************************

LINK_DIRECTORIES( "${PCSIM_SOURCE_DIR}/lib" )

ADD_LIBRARY( ${PCSIM_MODULE_NAME} SHARED
             ${MODULE_SOURCES} 
             ${SIM_OBJ_REG_CPP} )


TARGET_LINK_LIBRARIES(
  ${PCSIM_MODULE_NAME}
  	  ${DEP_MODULE_NAMES}	
	  pcsim
	  ${XERCES_LIBS} 
	  ${MPI_LIBS}
	  ${GSL_LIBS}
	  ${BOOST_THREAD} 
	  ${BOOST_DATETIME}
	  ${STLP_LIBS}
)

#**********************************************************************************************
#                            Building the Python extension module
#**********************************************************************************************

#
# make py<module>.so (p<module>.pyd)
#
FILE(GLOB BP_SOURCES "build/${MYPYPCSIM}/*pypp.cpp" )

ADD_LIBRARY( ${MYPYPCSIM} SHARED 
             ${BP_SOURCES} 
             ${MODPYPPMAIN}
             ${PCSIM_SOURCE_DIR}/python/pypcsim/warning_preventer.cpp )

ADD_DEPENDENCIES( ${MYPYPCSIM} ${PCSIM_MODULE_NAME} generate_code )

TARGET_LINK_LIBRARIES(
  ${MYPYPCSIM}	
	  ${PCSIM_MODULE_NAME}
	  pcsim	  
	  ${XERCES_LIBS}
	  ${MPI_LIBS}
	  ${GSL_LIBS}
	  ${BOOST_LIBS}
	  ${STLP_LIBS}
)

# ****************************************************************************************************
#                                   building CPPUNIT tests
# ****************************************************************************************************

FILE(GLOB TEST_SOURCES "cppunittests/*Test.cpp" )

REMOVE_DEFINITIONS( -Wall -Wnon-virtual-dtor )

ADD_EXECUTABLE( "${PCSIM_MODULE_NAME}_test" 
                "${PCSIM_SOURCE_DIR}/cppunittests/CppUnitMain.cpp" 
                ${TEST_SOURCES}                
                "${PCSIM_SOURCE_DIR}/cppunittests/commonUtils.cpp" )
                
ADD_DEPENDENCIES( "${PCSIM_MODULE_NAME}_test" ${PCSIM_MODULE_NAME} ) 

TARGET_LINK_LIBRARIES( "${PCSIM_MODULE_NAME}_test"
	  ${PCSIM_MODULE_NAME}	  
	  pcsim
	  ${XERCES_LIBS}
	  ${MPI_LIBS}
	  ${GSL_LIBS}
	  ${CPPUNIT_LIBS} 
	  ${BOOST_THREAD} 
	  ${BOOST_DATETIME}
	  ${STLP_LIBS}
)

ENABLE_TESTING()

ADD_TEST( LOCAL_CppUnitTests  "${EXECUTABLE_OUTPUT_PATH}/${PCSIM_MODULE_NAME}_test" AutoBuild )


IF( PLATFORM STREQUAL "windows-x86" )
  SET_TARGET_PROPERTIES( ${MYPYPCSIM} PROPERTIES PREFIX "" SUFFIX ".pyd" )
ELSE( PLATFORM STREQUAL "windows-x86" )
  SET_TARGET_PROPERTIES( ${MYPYPCSIM} PROPERTIES PREFIX "" )
ENDIF( PLATFORM STREQUAL "windows-x86" )

SUBDIRS( pyunit )
