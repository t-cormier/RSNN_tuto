

ENABLE_TESTING()
ADD_TEST( PyUnitTests  "${PYTHON_EXECUTABLE}" "${ABSCWD}/pyunit/run_all_tests.py"  "${ABSCWD}/pyunit" )

