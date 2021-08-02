#ifndef BIOFEEDREWARDGENTEST_H_
#define BIOFEEDREWARDGENTEST_H_

// includes from cppunit
#include <cppunit/TestFixture.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

// includes for our own extensions to cppunit
#include "CppUnitMain.h"

// include what ever you want to test
#include "BioFeedRewardGen.h"

class BioFeedRewardGenTest : public CppUnit::TestFixture
{

public:

    CPPUNIT_TEST_SUITE( BioFeedRewardGenTest );

    CPPUNIT_TEST( test_simple );
    
    CPPUNIT_TEST( test_random );

    CPPUNIT_TEST_SUITE_END();

    void test_simple();
    
    void test_random();
    
public:
    void setUp();
    void tearDown();

private :
    
};

#endif /*BIOFEEDREWARDGENTEST_H_*/
