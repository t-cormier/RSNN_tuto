#include "BioFeedRewardGen.h"

#include <boost/format.hpp>

using boost::format;


BioFeedRewardGen::~BioFeedRewardGen()
{}

int BioFeedRewardGen::reset(double dt)

{
    presynapticSpikes.reset( maxRelevantSpikeTimeDiff(dt) );
    postsynapticSpikes.reset( maxRelevantSpikeTimeDiff(dt) );
    new_reward = 0;
    reward = 0;
    return 0;
}

int BioFeedRewardGen::advance( AdvanceInfo const& )
{
    reward = new_reward ;
    new_reward = 0;
    return 0;
}

double BioFeedRewardGen::getAnalogOutput(analog_port_id_t port ) const
{
    return reward;
}

int BioFeedRewardGen::nSpikeInputPorts() const
{
    return 2;
}

int BioFeedRewardGen::nAnalogOutputPorts() const
{
    return 1;
}

SimObject::PortType BioFeedRewardGen::outputPortType(port_t p) const
{
    if (p == 0)
        return analog;
    return undefined;
}

SimObject::PortType BioFeedRewardGen::inputPortType(port_t p) const
{
    if (p < 2)
        return spiking;
    return undefined;
}

int BioFeedRewardGen::spikeHit( spikeport_t port, SpikeEvent const& spike )
{
    if( port == 0 ) {
        //cerr << "presynaptic spike arrived = " << spike.time() << endl;
        presynapticSpikes.insert( spike.time() );
        postsynapticSpikes.cutoff( spike.time() );
        preSpikeHit( spike );
        return 0;
    } else {
        //cerr << "postsynaptic spike arrived = " << spike.time() << endl;
        postsynapticSpikes.insert( spike.time() );
        presynapticSpikes.cutoff( spike.time() );
        postSpikeHit( spike );
        return 0;
    }
}

void BioFeedRewardGen::preSpikeHit( SpikeEvent const& pre_spike )
{
    double t_prev_pre = presynapticSpikes.second();

    if (postsynapticSpikes.size() > 1) {
        SpikeBuffer::const_iterator post_spike      = postsynapticSpikes.begin();
        SpikeBuffer::const_iterator post_spike_end  = postsynapticSpikes.end_of_window();
        SpikeBuffer::const_iterator prev_post_spike = post_spike;
        for (prev_post_spike++; prev_post_spike != post_spike_end ; post_spike++, prev_post_spike++ ) {
            calculateReward( *post_spike - pre_spike.time(), *post_spike, pre_spike.time(), *prev_post_spike, t_prev_pre );
        }
    }

}


void BioFeedRewardGen::postSpikeHit( SpikeEvent const& post_spike )
{
    double t_prev_post = postsynapticSpikes.second();

    if (presynapticSpikes.size() > 1) {
        SpikeBuffer::const_iterator pre_spike      = presynapticSpikes.begin();
        SpikeBuffer::const_iterator pre_spike_end  = presynapticSpikes.end_of_window();

        SpikeBuffer::const_iterator prev_pre_spike = pre_spike;
        for (prev_pre_spike++; prev_pre_spike != pre_spike_end ; pre_spike++, prev_pre_spike++ ) {
            calculateReward( post_spike.time() - *pre_spike, post_spike.time(), *pre_spike, t_prev_post, *prev_pre_spike );
        }
    }

}

double BioFeedRewardGen::calculateReward(const double & delta, const double  & t_post, const double & t_pre, const double & t_prev_post, const double & t_prev_pre )
{
    double d_reward = 0.0;
    //cerr << format("inside learning delta=%10f ") % delta << " t_post=" << t_post << "t_pre=" << t_pre << endl;
    if (delta < - Gap - Te) {
        d_reward = Aneg * (- delta < tauneg + Te ? 1.0 : 0.0);
    } else if (delta > Gap - Te) {
        d_reward = Apos * ( delta < taupos - Te ? 1.0 : 0.0);
    }
    new_reward += d_reward;
    return d_reward;
}

double BioFeedRewardGen::maxRelevantSpikeTimeDiff(double dt)
{
    return std::max(tauneg + dt + Te, taupos + dt + Te);
}
