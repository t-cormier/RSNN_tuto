#include "ReadoutRewardGen.h"

ReadoutRewardGen::~ReadoutRewardGen()
{
}

int ReadoutRewardGen::reset(double dt)

{
	hadSpike = false;
	reward = 0;
	return 0;
}

void ReadoutRewardGen::setAnalogInput(double value, analog_port_id_t port)
{
	scale = value;
}

double ReadoutRewardGen::getAnalogOutput(analog_port_id_t port) const
{
	return reward;
}

int ReadoutRewardGen::nSpikeInputPorts() const
{
	return 1;
}

int ReadoutRewardGen::nAnalogOutputPorts() const
{
	return 1;
}

SimObject::PortType ReadoutRewardGen::outputPortType(port_t p) const
{
	if (p == 0)
		return analog;
	return undefined;
}

SimObject::PortType ReadoutRewardGen::inputPortType(port_t p) const
{
	if (p == 0)
		return spiking;
	else if (p == 1)
		return analog;

	return undefined;
}

int ReadoutRewardGen::advance(AdvanceInfo const&)
{
	if (hadSpike)
		reward = scale;
	else
		reward = 0;
	hadSpike = false;
	return 0;
}

int ReadoutRewardGen::spikeHit(spikeport_t port, SpikeEvent const& spike)
{
	hadSpike = true;
	return 0;
}
