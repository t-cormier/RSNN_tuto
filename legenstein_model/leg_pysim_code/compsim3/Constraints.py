from numpy import *

def KappaFunction(t,KappaApos, KappaAneg, KappaTaupos, KappaTauneg, KappaTe, KappaTaupos2  = 3e-3, KappaTauneg2 = 3e-3, KernelType = 'square'):
    if KernelType == 'DblExp':
        if t + KappaTe > 0:
            return KappaApos * ( exp(- (t + KappaTe) / KappaTaupos ) - exp( - (t + KappaTe)/KappaTaupos2) )
        else:
            return KappaAneg * ( exp(- (-t - KappaTe) / KappaTauneg ) - exp( - (- t - KappaTe)/KappaTauneg2) )
    else:
        if t > KappaTaupos - KappaTe:
            return 0
        elif t > - KappaTe:
            return KappaApos
        elif t > - KappaTauneg - KappaTe:
            return KappaAneg
        return 0
    return 0


def optimal_Te_value(maxValue,TeDT,KappaApos,KappaAneg,KappaTaupos,KappaTauneg,KappaTaupos2,KappaTauneg2,KernelType,synTau):
    DT = 1e-5    
    LeftMargin = -0.3
    RightMargin = 0.3 + DT
    EpsilonArray = array( [ 0 for t in arange(LeftMargin, 0, DT)] + [ 1/synTau * exp(-t/synTau) for t in arange(0, RightMargin, DT) ])        
    EpsilonArray = EpsilonArray[::-1]
    KappaArray = array([ KappaFunction(t, KappaApos, KappaAneg, KappaTaupos, KappaTauneg, 0, KappaTaupos2, KappaTauneg2, KernelType) for t in arange(LeftMargin, RightMargin+maxValue, DT) ])
    prevSign = '1'
    for KappaTe in arange(0,maxValue,TeDT):
        Sign = sum(EpsilonArray * KappaArray[int(KappaTe/DT):int(KappaTe/DT)+len(EpsilonArray)]) * DT > 0
        if not prevSign == '1':
            if not prevSign == Sign:
                return KappaTe
        prevSign = Sign
    

def WStdpFunction(t,stdpApos,stdpAneg, stdpTaupos, stdpTauneg):    
    if t >= 0:
        return stdpApos * exp(-t/stdpTaupos)
    else:
        return stdpAneg * exp(t/stdpTauneg)


def checkConstraints(synTau, NumSyn, ratioStrong, Wmax, inputRate, Rbase, 
                     stdpApos, stdpAneg, stdpTaupos, stdpTauneg, 
                     DAStdpRate, DATraceDelay, DATraceTau, DATraceShape, rewardDelay, 
                     KappaApos, KappaAneg, KappaTaupos, KappaTauneg, KappaTe, numAdditionalTargetSynapses, KappaTaupos2 = 3e-3, KappaTauneg2 = 3e-3, KernelType = 'square'):
    if stdpTaupos == synTau:
        synTau = synTau * 1.0001
    if stdpTauneg == synTau:
        synTau = synTau * 1.0001

    DT = 1e-5
    
    LeftMargin = -0.3
    RightMargin = 0.3 + DT
    
    KappaArray = array([ KappaFunction(t, KappaApos, KappaAneg, KappaTaupos, KappaTauneg, KappaTe, KappaTaupos2, KappaTauneg2, KernelType) for t in arange(LeftMargin, RightMargin, DT) ])
    
    EpsilonArray = array( [ 0 for t in arange(LeftMargin, 0, DT)] + [ 1/synTau * exp(-t/synTau) for t in arange(0, RightMargin, DT) ])
    
    WstdpArray = array([ WStdpFunction(t,stdpApos,stdpAneg, stdpTaupos, stdpTauneg) for t in arange(LeftMargin, RightMargin, DT) ])
    
    EpsilonKappaArray = convolve(KappaArray, EpsilonArray, 'same') * DT
    
    IntWstdpEpsKappaNumeric = sum(WstdpArray * EpsilonKappaArray) * DT
    
    print "IntWstdpEpsKappaNumeric = ", IntWstdpEpsKappaNumeric
    
    IntWstdpEpsilonEpsKappaNumeric = sum(WstdpArray * EpsilonArray * EpsilonKappaArray) * DT
    
    print "IntWstdpEpsilonEpsKappaNumeric = ", IntWstdpEpsilonEpsKappaNumeric
    
    IntEpsilonEpsilonKappaNumeric = sum(EpsilonArray * EpsilonKappaArray) * DT
    
    print "IntEpsilonEpsilonKappaNumeric = ", IntEpsilonEpsilonKappaNumeric
    
    IntWstdpEpsilonNumeric = sum(WstdpArray * EpsilonArray) * DT
    
    print "IntWstdpEpsilonNumeric = ", IntWstdpEpsilonNumeric
    
    IntKappaNumeric = sum(KappaArray) * DT
    
    print "IntKappaNumeric = ", IntKappaNumeric
    
    IntWstdpNumeric = sum(WstdpArray) * DT
    
    print "IntWstdpNumeric = ", IntWstdpNumeric
    
    print "Rbase = ", Rbase
    print "Wmax = ", Wmax
    print "NumSyn = ", NumSyn
    print "inputRate = ", inputRate
    
    
    Rpost = Rbase + Wmax * NumSyn * inputRate * 1.0 / 2.0
    
    print "Rpost = ", Rpost
    
    Rstar = Wmax * ratioStrong * (NumSyn + numAdditionalTargetSynapses) * inputRate
    
    print "Rstar = ", Rstar
    
    Weight = Wmax / 2
    
    VarianceFactor = Rstar * Rpost * KappaTaupos * stdpApos * 1e5
    
    
    if DATraceShape == 'exp':
        FcTr = exp(- (rewardDelay - DATraceDelay) / DATraceTau )
    elif DATraceShape == 'alpha':
        FcTr = (rewardDelay - DATraceDelay) / DATraceTau * exp( - (rewardDelay - DATraceDelay) / DATraceTau )    
    
    IntFc = DATraceTau    
    
    
    Term1 = IntKappaNumeric * ( Rpost * IntWstdpNumeric + Weight * IntWstdpEpsilonNumeric ) * ( Rpost * Rstar * IntFc +  FcTr * (Rstar + Rstar * Weight +  Rpost * Wmax))
    
    Term2 = FcTr * Wmax * Rpost * IntWstdpEpsKappaNumeric
    
    Term3 = FcTr * Wmax * Weight * IntWstdpEpsilonEpsKappaNumeric
    
    Term4 = FcTr * Wmax * Weight * Rpost * IntWstdpNumeric * IntEpsilonEpsilonKappaNumeric
    
    Term5 = FcTr * Wmax * Weight * Weight * IntWstdpEpsilonNumeric * IntEpsilonEpsilonKappaNumeric
    
    print "Term1 = ", Term1
    print "Term2 = ", Term2
    print "Term3 = ", Term3
    print "Term4 = ", Term4
    print "Term5 = ", Term5
    
    
    DW_over_DT_strong = Term1 + Term2 + Term3 + Term4 + Term5
    
    DW_over_DT_strong = DAStdpRate * inputRate * DW_over_DT_strong
    
    
    print "********************************************************"    
    print "DW_over_DT_strong = ", DW_over_DT_strong
    
    Tconv_strong = Wmax / 2.0 / DW_over_DT_strong
    
    print "Tconv strong = ", Tconv_strong
    
    print "Tconv_strong in hours = ", Tconv_strong / 3600
    
    
    DW_over_DT_weak = DAStdpRate * IntKappaNumeric * IntFc * Rstar * inputRate * Rpost * ( Rpost * IntWstdpNumeric + Weight * IntWstdpEpsilonNumeric ) +\
                      IntKappaNumeric * FcTr * inputRate * ( Rpost * IntWstdpNumeric + Weight * IntWstdpEpsilonNumeric ) * ( Rstar + Rstar * Weight )
                      
    
    print "DW_over_DT_weak = ", DW_over_DT_weak    
    
    Tconv_weak = Wmax / 2 / -DW_over_DT_weak
    
    print "Tconv_weak = ", Tconv_weak
    
    print "Tconv_weak in hours = ", Tconv_weak / 3600
                                                
    IntWEpsKappa1 = (-stdpAneg) * (-KappaAneg) * stdpTauneg * exp(-KappaTe * stdpTauneg ) 
    IntWEpsKappa2 = (-stdpAneg) * KappaApos * (  stdpTauneg * synTau / (stdpTauneg - synTau) * (exp(KappaTe * (stdpTauneg - synTau) / (stdpTauneg * synTau) ) - 1 ) +\
                     stdpTauneg * (exp(-KappaTe / stdpTauneg ) - 1) )
    IntWEpsKappa3 = stdpApos * KappaApos * ( stdpTaupos - stdpTaupos * synTau / (stdpTaupos + synTau)  )
    
    IntWEpsKappa = IntWEpsKappa1 + IntWEpsKappa2 + IntWEpsKappa3
                                                
    print "IntWepsKappa = ", IntWEpsKappa                                            
                                                
    IntWstdp = abs(stdpApos * stdpTaupos + stdpAneg * stdpTauneg)
    
    print "IntWstdp = ", IntWstdp
    
    IntWstdpEpsil = stdpApos * stdpTaupos / (synTau + stdpTaupos)
    
    print "IntWstdpEpsil", IntWstdpEpsil
    
    IntKappa = KappaApos * KappaTaupos + KappaAneg * KappaTauneg
    
    print "IntKappa = ", IntKappa
    
    if DATraceShape == 'exp':
        IntFcOverFcTr = DATraceTau / exp(- (rewardDelay - DATraceDelay) / DATraceTau )
    elif DATraceShape == 'alpha':
        IntFcOverFcTr = DATraceTau / (rewardDelay - DATraceDelay) * exp( - (rewardDelay - DATraceDelay) / DATraceTau )
        
    
    RatePostMax = Rbase + NumSyn * Wmax * inputRate
    
    minLearningNrnRate = Rbase + Wmax * NumSyn * (1.0 / 4.0) * inputRate
    
    targetNrnRate = Wmax * NumSyn * ratioStrong * inputRate
    
    
    print "********************************************************"
    
    
    LeftSide_weak = minLearningNrnRate * abs(IntWstdpNumeric)
    RightSide_weak = IntWstdpEpsil * Wmax
    
    
    LeftSide_strong = IntWstdpEpsKappaNumeric
    RightSide_strong = abs(IntWstdpNumeric) * IntKappaNumeric * targetNrnRate / Wmax * ( Rpost * IntFcOverFcTr  + 1)
    
    
    return [ DW_over_DT_strong , DW_over_DT_weak, Tconv_strong, Tconv_weak, LeftSide_weak, RightSide_weak, LeftSide_strong, RightSide_strong ]

