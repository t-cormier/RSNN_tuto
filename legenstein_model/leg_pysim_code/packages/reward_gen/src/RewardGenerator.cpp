#include "RewardGenerator.h"


RewardGenerator::RewardGenerator(double Rduration, double Rnegbase, bool AvgRewardZero, bool OneOverRateScale)
        : rewardDuration(Rduration), Rnegbase(Rnegbase), AvgRewardZero(AvgRewardZero), OneOverRateScale(OneOverRateScale)
{}

RewardGenerator::~RewardGenerator()
{
}

int RewardGenerator::reset(double dt)
{   
    A = 1;
    stepsLeftForReward = int(rewardDuration / dt);
    lastSpikeTime = - stepsLeftForReward * dt - 2 * dt;
    stepsLeftForReward = 0;
    reward = Rnegbase;
    instRate = 0;
    isActive = true;
    return 0;
}

int RewardGenerator::advance( AdvanceInfo const& )
{   
    if (stepsLeftForReward > 0)
    	reward = A;
    else
    	reward = Rnegbase;
	stepsLeftForReward = stepsLeftForReward - 1;
	if (!isActive)
		reward = Rnegbase;    	    
    return 0;
}

int RewardGenerator::spikeHit( spikeport_t port, SpikeEvent const& spike )
{
	if (stepsLeftForReward > 0)
		return 0;
    stepsLeftForReward = int(rewardDuration / spike.dt.in_sec());
    if (AvgRewardZero)
    	if (OneOverRateScale)
    		A = ((1/instRate) - spike.dt.in_sec()) * fabs(Rnegbase) / (stepsLeftForReward * spike.dt.in_sec());
    	else
    		A = (spike.t - lastSpikeTime - (stepsLeftForReward * spike.dt.in_sec())) * fabs(Rnegbase) / (stepsLeftForReward * spike.dt.in_sec());
    lastSpikeTime = spike.t;
    return 0;
}

double RewardGenerator::getAnalogOutput(analog_port_id_t port ) const
{
    return reward;
}

//! Analog input to given port
void RewardGenerator::setAnalogInput(double value, analog_port_id_t port)
{
	instRate = value;
}

int RewardGenerator::nSpikeInputPorts() const
{
    return 1;
}


int RewardGenerator::nAnalogInputPorts() const
{
    return 1;
}


int RewardGenerator::nAnalogOutputPorts() const
{
    return 1;
}

SimObject::PortType RewardGenerator::outputPortType(port_t p) const
{
    if (p == 0)
        return analog;
    return undefined;
}

SimObject::PortType RewardGenerator::inputPortType(port_t p) const
{
    if (p == 0)
        return spiking;
    else if (p == 1)
    	return analog;        
    return undefined;
}
