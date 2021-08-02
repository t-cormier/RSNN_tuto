#ifndef RewardGenerator2_H_
#define RewardGenerator2_H_

#include "SimObject.h"
#include "SpikeSender.h"

class RewardGenerator2 : public SimObject, public SingleOutputSpikeSender
{
	SIMOBJECT( RewardGenerator2, AdvancePhase::One )
public:
	
	RewardGenerator2();
	   
	virtual ~RewardGenerator2();
	    
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
    
    double instRate1;
    
    double instRate2;    
    
};

#endif /*RewardGenerator2_H_*/
