#ifndef READOUTREWARDGEN_H_
#define READOUTREWARDGEN_H_

#include "SimObject.h"

class ReadoutRewardGen : public SimObject {
	SIMOBJECT( ReadoutRewardGen, AdvancePhase::One )
public:

	ReadoutRewardGen()
	{};

	virtual ~ReadoutRewardGen();

	virtual int reset(double dt);

	virtual int advance( AdvanceInfo const& );

	virtual int spikeHit( spikeport_t port, SpikeEvent const& spike );

	virtual void setAnalogInput(double value, analog_port_id_t port = 0);

	virtual double getAnalogOutput(analog_port_id_t port = 0) const;

	virtual int nSpikeInputPorts() const;

	virtual int nAnalogOutputPorts() const;

	virtual PortType outputPortType(port_t p) const;

	virtual PortType inputPortType(port_t p) const;

	//! The generated reward [readonly; range=(-1e6,1e6);]
	double reward;

	//! Scale factor of the incoming spikes
	double scale;

private:
	bool hadSpike;
};

#endif /*READOUTREWARDGEN_H_*/
