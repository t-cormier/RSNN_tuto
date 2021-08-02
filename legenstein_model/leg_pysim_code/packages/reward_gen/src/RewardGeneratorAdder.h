#ifndef RewardGeneratorAdder_H_
#define RewardGeneratorAdder_H_

#include "SimObject.h"
#include "SpikeSender.h"

class RewardGeneratorAdder : public SimObject, public SingleOutputSpikeSender
{
	SIMOBJECT( RewardGeneratorAdder, AdvancePhase::One )
public:
	
	RewardGeneratorAdder();
	   
	virtual ~RewardGeneratorAdder();
	    
    virtual int reset(double dt);
            
    virtual int advance( AdvanceInfo const& );
            
    virtual int spikeHit( spikeport_t port, SpikeEvent const& spike );
    
    virtual double getAnalogOutput(analog_port_id_t port = 0) const;
    
    virtual void setAnalogInput(double value, analog_port_id_t port = 0);
    
    virtual int nSpikeInputPorts() const;  
      
    virtual int nAnalogOutputPorts() const;
    
    virtual int nAnalogInputPorts() const;
    
    virtual PortType outputPortType(port_t p) const;
    
    virtual PortType inputPortType(port_t p) const;
    
    bool isActive;

protected:
	double reward;    
    
    double rateSum;
    
};

#endif /*RewardGeneratorAdder_H_*/
