#include "BioFeedRewardGenTest.h"

#include "SingleThreadNetwork.h"
#include "SpikingInputNeuron.h"
#include "AnalogRecorder.h"
#include "RandomDistribution.h"


#include <iostream>
using std::cerr;
using std::endl;

// choose one from the following
PCSIM_LOCAL_AUTOBUILD_TEST_SUITE( BioFeedRewardGenTest )
// PCSIM_LOCAL_NIGHTLY_TEST_SUITE( BioFeedRewardGenTest )
// PCSIM_MPI_AUTOBUILD_TEST_SUITE( BioFeedRewardGenTest )
// PCSIM_MPI_NIGHTLY_TEST_SUITE( BioFeedRewardGenTest )

void BioFeedRewardGenTest::setUp()
{}

void BioFeedRewardGenTest::tearDown()
{}

void BioFeedRewardGenTest::test_simple()
{
    SingleThreadNetwork net;
    double Tsim = 1.5;
    double dt = 1e-4;

    SimObject::ID::Packed learn_nrn = net.add( SpikingInputNeuron() );
    SimObject::ID::Packed target_nrn = net.add( SpikingInputNeuron() );
    SimObject::ID::Packed rec = net.add( AnalogRecorder() );
    SimObject::ID::Packed reward_gen = net.add( BioFeedRewardGen( 0.5, -0.5, 60e-3, 40e-3, 2e-3, 0.0 ) );

    net.connect(learn_nrn, 0, reward_gen, 1, Time::sec(0));
    net.connect(target_nrn, 0, reward_gen, 0, Time::sec(0));
    net.connect(reward_gen, rec, Time::sec(0));

    vector<double> learn_nrn_spikes;
    vector<double> target_nrn_spikes;

    target_nrn_spikes.push_back( 0.0010 );

    target_nrn_spikes.push_back( 0.2400 );
    target_nrn_spikes.push_back( 0.2411 );
    target_nrn_spikes.push_back( 0.2490 );

    target_nrn_spikes.push_back( 0.4010 );

    target_nrn_spikes.push_back( 0.6200 );
    target_nrn_spikes.push_back( 0.6290 );
    target_nrn_spikes.push_back( 0.6405 );
    target_nrn_spikes.push_back( 0.6510 );

    target_nrn_spikes.push_back( 0.8520 );


    learn_nrn_spikes.push_back( 0.0230 );
    learn_nrn_spikes.push_back( 0.0650 );
    learn_nrn_spikes.push_back( 0.2010 );

    learn_nrn_spikes.push_back( 0.4025 );

    learn_nrn_spikes.push_back( 0.6852 );

    learn_nrn_spikes.push_back( 0.8100 );
    learn_nrn_spikes.push_back( 0.8210 );
    learn_nrn_spikes.push_back( 0.8273 );
    learn_nrn_spikes.push_back( 0.8480 );

    dynamic_cast<SpikingInputNeuron *>(net.object(learn_nrn))->setSpikes(learn_nrn_spikes);
    dynamic_cast<SpikingInputNeuron *>(net.object(target_nrn))->setSpikes(target_nrn_spikes);

    net.reset();
    net.advance( int( Tsim/dt ) );

    vector<double > expectedOutValues( int( Tsim/dt ) );
    const vector<double> & actualOutValues = dynamic_cast<AnalogRecorder *>(net.object(rec))->getRecordedValues();

    expectedOutValues[ 231 ] = 0.5;

    expectedOutValues[ 2401 ] = - 0.5;

    expectedOutValues[ 6853 ] = 1.5;

    expectedOutValues[ 8521 ] = - 1.5;

    CPPUNIT_ASSERT_EQUAL( expectedOutValues.size(), actualOutValues.size() );

    for (unsigned i = 0; i < (unsigned)expectedOutValues.size() ; ++i) {
        CPPUNIT_ASSERT_EQUAL( expectedOutValues[i], actualOutValues[i] );
    }
}

void BioFeedRewardGenTest::test_random()
{
    SingleThreadNetwork net;
    double Tsim = 1.5;
    double dt = 1e-4;

    double Apos = 0.5;
    double Aneg = -0.5;
    double taupos = 60e-3;
    double tauneg = 40e-3;
    double gap = 2e-3;

    SimObject::ID::Packed learn_nrn = net.add( SpikingInputNeuron() );
    SimObject::ID::Packed target_nrn = net.add( SpikingInputNeuron() );
    SimObject::ID::Packed rec = net.add( AnalogRecorder() );
    SimObject::ID::Packed reward_gen = net.add( BioFeedRewardGen( Apos, Aneg, taupos, tauneg, gap,  0.0 ) );

    net.connect(learn_nrn, 0, reward_gen, 1, Time::sec(0));
    net.connect(target_nrn, 0, reward_gen, 0, Time::sec(0));
    net.connect(reward_gen, rec, Time::sec(0));

    vector<double> learn_nrn_spikes;
    vector<double> target_nrn_spikes;

    ExponentialDistribution exprnd( 50 );
    MersenneTwister19937 rndeng;
    rndeng.seed( 1234508 );
    double spike_time = 0;
    while (spike_time < Tsim ) {
        spike_time += exprnd( rndeng ) + dt;
        if (spike_time < Tsim)
           target_nrn_spikes.push_back( spike_time );
    }

    spike_time = 0;
    while (spike_time < Tsim ) {
        spike_time += exprnd( rndeng) + dt;
        if (spike_time < Tsim)
            learn_nrn_spikes.push_back( spike_time );
    }

    dynamic_cast<SpikingInputNeuron *>(net.object(learn_nrn))->setSpikes(learn_nrn_spikes);
    dynamic_cast<SpikingInputNeuron *>(net.object(target_nrn))->setSpikes(target_nrn_spikes);

    /* cerr << " learn_nrn_spikes : " << endl;
    dynamic_cast<SpikingInputNeuron *>(net.object(learn_nrn))->printSpikeTimes();
    cerr << " target_nrn_spikes : " << endl;
    dynamic_cast<SpikingInputNeuron *>(net.object(target_nrn))->printSpikeTimes(); */

    net.reset();
    net.advance( int( Tsim/dt ) );

    vector<double > expectedOutValues( int( Tsim/dt ) );

    // calculate expected values
    for ( unsigned i = 0; i < (unsigned)learn_nrn_spikes.size(); ++i ) {
        for ( unsigned j = 0; j < (unsigned)target_nrn_spikes.size() ; ++j ) {
            double delta = learn_nrn_spikes[i] - target_nrn_spikes[j];
            if (delta > 0) {
                unsigned coor = unsigned( learn_nrn_spikes[i] / dt ) + 2;
                if (delta > gap && delta < taupos)
                    expectedOutValues[coor] += Apos;
            } else {
                unsigned coor = unsigned( target_nrn_spikes[j] / dt ) + 2;
                if ( -delta > gap && -delta < tauneg )
                    expectedOutValues[coor] += Aneg;
            }
        }
    }

    const vector<double> & actualOutValues = dynamic_cast<AnalogRecorder *>(net.object(rec))->getRecordedValues();

    CPPUNIT_ASSERT_EQUAL( expectedOutValues.size(), actualOutValues.size() );

    for (unsigned i = 0; i < (unsigned)expectedOutValues.size() ; ++i) {
        try {
            //cerr << "values " << i << " expected = " << expectedOutValues[i] << " actual=" << actualOutValues[i] << endl;
            CPPUNIT_ASSERT_EQUAL( expectedOutValues[i], actualOutValues[i] );
        } catch ( CppUnit::Exception & e) {
            cerr << "values " << i << " " << endl;
            i++;
            cerr << "values " << i << " expected = " << expectedOutValues[i] << " actual=" << actualOutValues[i] << endl;
            throw( e );
        }
    }

}

