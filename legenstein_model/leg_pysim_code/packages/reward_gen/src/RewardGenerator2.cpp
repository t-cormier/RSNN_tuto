#include "RewardGenerator2.h"

RewardGenerator2::RewardGenerator2()        
{}

RewardGenerator2::~RewardGenerator2()
{
}

int RewardGenerator2::reset(double dt)
{   
    instRate1 = 0;
    instRate2 = 0;    
    return 0;
}

int RewardGenerator2::advance( AdvanceInfo const& )
{   
    reward = instRate1 + instRate2;
    return 0;
}

int RewardGenerator2::spikeHit( spikeport_t port, SpikeEvent const& spike )
{		
    return 0;
}

double RewardGenerator2::getAnalogOutput(analog_port_id_t port ) const
{
    return reward;
}

//! Analog input to given port
void RewardGenerator2::setAnalogInput(double value, analog_port_id_t port)
{
	if (port == 1)
		instRate1 = value;
	else	
		instRate2 = value;
}

int RewardGenerator2::nSpikeInputPorts() const
{
    return 1;
}


int RewardGenerator2::nAnalogInputPorts() const
{
    return 1;
}


int RewardGenerator2::nAnalogOutputPorts() const
{
    return 1;
}

SimObject::PortType RewardGenerator2::outputPortType(port_t p) const
{
    if (p == 0)
        return analog;
    return undefined;
}

SimObject::PortType RewardGenerator2::inputPortType(port_t p) const
{
    if (p == 0)
        return spiking;
    else if (p < 3)
    	return analog;        
    return undefined;
}
