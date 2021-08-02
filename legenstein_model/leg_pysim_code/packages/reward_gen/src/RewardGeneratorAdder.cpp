#include "RewardGeneratorAdder.h"

RewardGeneratorAdder::RewardGeneratorAdder()        
{}

RewardGeneratorAdder::~RewardGeneratorAdder()
{
}

int RewardGeneratorAdder::reset(double dt)
{   
    rateSum = 0;
    reward = 0;
    return 0;
}

int RewardGeneratorAdder::advance( AdvanceInfo const& )
{   
    reward = rateSum;
    rateSum = 0;
    return 0;
}

int RewardGeneratorAdder::spikeHit( spikeport_t port, SpikeEvent const& spike )
{		
    return 0;
}

double RewardGeneratorAdder::getAnalogOutput(analog_port_id_t port ) const
{
    return reward;
}

//! Analog input to given port
void RewardGeneratorAdder::setAnalogInput(double value, analog_port_id_t port)
{
	rateSum += value;
}

int RewardGeneratorAdder::nSpikeInputPorts() const
{
    return 0;
}


int RewardGeneratorAdder::nAnalogInputPorts() const
{
    return 1000;
}


int RewardGeneratorAdder::nAnalogOutputPorts() const
{
    return 1;
}

SimObject::PortType RewardGeneratorAdder::outputPortType(port_t p) const
{
    if (p < 1)
    	return analog;
    return undefined;
}

SimObject::PortType RewardGeneratorAdder::inputPortType(port_t p) const
{   
	if (p < 1000)
    	return analog;        
    return undefined;
}
