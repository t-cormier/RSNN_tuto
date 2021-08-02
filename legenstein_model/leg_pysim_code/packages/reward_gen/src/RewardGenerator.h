#ifndef REWARDGENERATOR_H_
#define REWARDGENERATOR_H_

#include "SimObject.h"
#include "SpikeSender.h"

class RewardGenerator : public SimObject, public SingleOutputSpikeSender
{
	SIMOBJECT( RewardGenerator, AdvancePhase::One )
public:
	
	RewardGenerator(double Rduration = 5, 
					double Rnegbase = 0.0, 
					bool AvgRewardZero = false, 
					bool OneOverRateScale = false);
	   
	virtual ~RewardGenerator();
	    
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
    int stepsLeftForReward;
    double rewardDuration;
    double Rnegbase;
    double A;
    bool AvgRewardZero;
    double lastSpikeTime;
    
    bool OneOverRateScale;
    
    double instRate;    
    
};

#endif /*REWARDGENERATOR_H_*/
