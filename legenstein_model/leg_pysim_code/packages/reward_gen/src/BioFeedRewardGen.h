#ifndef BIOFEEDREWARDGEN_H_
#define BIOFEEDREWARDGEN_H_

#include "SimObject.h"
#include "SpikeSender.h"
#include "SpikeBuffer.h"

class BioFeedRewardGen : public SimObject, public SingleOutputSpikeSender
{
    SIMOBJECT( BioFeedRewardGen, AdvancePhase::One )
public:

    BioFeedRewardGen(float Apos = 1,
                     float Aneg = -1,
                     float taupos = 60e-3,
                     float tauneg = 40e-3,
                     float Gap = 2e-3, 
                     float Te = 0.8e-3) :
            Gap(Gap), taupos(taupos), tauneg(tauneg), Apos(Apos), Aneg(Aneg), Te(Te)
    {
    }
    ;

    virtual ~BioFeedRewardGen();

    virtual int reset(double dt);

    virtual int advance( AdvanceInfo const& );

    virtual int spikeHit( spikeport_t port, SpikeEvent const& spike );

    virtual double getAnalogOutput(analog_port_id_t port = 0) const;

    virtual int nSpikeInputPorts() const;

    virtual int nAnalogOutputPorts() const;

    virtual PortType outputPortType(port_t p) const;

    virtual PortType inputPortType(port_t p) const;


    //! Called for each pre and post spike with time difference delta=t_post-t_pre
    double calculateReward(const double & delta, const double  & t_post, const double & t_pre, const double & t_prev_post, const double & t_prev_pre ) ;

    virtual double maxRelevantSpikeTimeDiff(double dt);

    //! Called if the pre-synaptic spikes hits the synapse and calls the learning function for each pair of pre-post spikes.
    void preSpikeHit( SpikeEvent const& spike );

    //! Called if the post-synaptic spikes hits the synapse and calls the learning function for each pair of post-pre spikes.
    void postSpikeHit( SpikeEvent const& spike );


    //! The generated reward [readonly; range=(-1e6,1e6);]
    double reward;

    //! The time interval where reward is ineffective [range=(0,1)]
    float Gap;

    //! The width of the positive part of the Kappa kernel [range=(0,1);]
    float taupos;

    //! The width of the negative part of the Kappa kernel [range=(0,1);]
    float tauneg;

    //! The height of the positive part of the Kappa kernel [range=(0,1);]
    float Apos;

    //! The height of the negative part of the Kappa kernel [range=(0,1);]
    float Aneg;
    
    
    //! The synaptic delay, which causes shift of the kernel to the left
    float Te;


private:
    double new_reward;

    SpikeBuffer postsynapticSpikes;
    SpikeBuffer presynapticSpikes;
};

#endif /*BIOFEEDREWARDGEN_H_*/
