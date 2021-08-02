import sys
import os
import re
sys.path.append('../packages')
from numpy import *
import random, getopt
from datetime import datetime
from math import *
from pylab import *
from tables import *
from math import exp 
from BeforeAfterExperiment import *
from frame import FrameAxes

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)



dummy_net = SingleThreadNetwork()

 

def display_spike_trains(f, idx, XBeforeMin, XBeforeMax, ax):
    r = constructRecordingsFromH5File(f)
        
    errorbar( (r.before_learning_spikes[idx] - XBeforeMin), 7*ones(len(r.before_learning_spikes[idx])), 0.8 * ones(len(r.before_learning_spikes[idx])), capsize = 0, visible = False, color = 'k')
    errorbar( (r.target_nrn_spikes[idx] - XBeforeMin), 5*ones(len(r.target_nrn_spikes[idx])), 0.8 * ones(len(r.target_nrn_spikes[idx])), capsize = 0, visible = False, color = 'k')
    errorbar( (r.realiz_target_nrn_spikes[idx] - XBeforeMin), 3*ones(len(r.realiz_target_nrn_spikes[idx])), 0.8 * ones(len(r.realiz_target_nrn_spikes[idx])), capsize = 0, visible = False, color = 'k')
    errorbar( (r.after_learning_spikes[idx] - XBeforeMin), ones(len(r.after_learning_spikes[idx])), 0.8 * ones(len(r.after_learning_spikes[idx])), capsize = 0, visible = False, color = 'k')
    
    xlim(0, XBeforeMax - XBeforeMin + 0.04)     
    xlabel("time [sec]")    
    ylim(0,8)
    yticks([])    
    
    text(-0.27, 0.871,'before learning', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    
    text(-0.27, 0.715,'target $S^*$', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    text(-0.27, 0.625,'(= rewarded', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    text(-0.27, 0.535,'spike times)', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    
    text(-0.27, 0.385,'realizable part', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    
    text(-0.27, 0.298,'of target $S^*$', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)
    
    text(-0.27, 0.125,'after learning', horizontalalignment='center',
         verticalalignment='center', fontsize = 13, transform = ax.transAxes)


def norm_vec(v):
     return sqrt(dot(v,v))
 
def rectangle_kernel(x, width):
    if x < width or x > -width:
        return 1
    return 0
    
def calculate_corr_coeff(spikes, target_spikes, start, end):        
    sigma = 2e-3
    kernel = lambda x:  rectangle_kernel( x, sigma)
         
    kernel_spikes = convolve_spikes(spikes, kernel, 1e-3, start, end, -sigma*10, sigma*10) 
    target_kernel = convolve_spikes(target_spikes, kernel, 1e-3, start, end, -sigma*10, sigma*10)
      
    corr_coeff = dot(kernel_spikes, target_kernel) / ( norm_vec(kernel_spikes) * norm_vec(target_kernel) )
    
    
    return corr_coeff

def generate_spike_corr(h5file, XBeforeMin, XBeforeMax):
    r = constructRecordingsFromH5File(h5file)
    corr = []
    for i in range(len(r.before_learning_spikes)):
        s = clip_window(r.before_learning_spikes[i], XBeforeMin, XBeforeMax, shift = True)
        target_s = clip_window(r.realiz_target_nrn_spikes[i], XBeforeMin, XBeforeMax, shift = True)
        corr.append(calculate_corr_coeff(s, target_s, 0, XBeforeMax - XBeforeMin ))
    after_learn_s = clip_window(r.after_learning_spikes[0], XBeforeMin, XBeforeMax, shift = True)
    target_s =  clip_window(r.realiz_target_nrn_spikes[0], XBeforeMin, XBeforeMax, shift = True)   
    corr.append(calculate_corr_coeff(after_learn_s, target_s, 0, XBeforeMax - XBeforeMin ))
    return  corr
    
def multi_run_and_save_beforeAfter(results_file, h5file, numRuns, Tsim):
    new_rec = Recordings(dummy_net)
    new_rec.before_learning_spikes = []
    new_rec.after_learning_spikes = []
    new_rec.target_nrn_spikes = []
    new_rec.realiz_target_nrn_spikes = []
    for i in range(numRuns):
        sampleIdx = i * int(200/numRuns)
        exper = BeforeAfterExperiment('beforeAfter', experParams = {"Tsim" : Tsim}, modelParams = {"biofeed": {"sampleIdx":sampleIdx, "h5filename" : results_file}})         
        exper.run("longrun")
        r = constructRecordingsFromH5File(exper.data_filename).biofeed
        os.remove(exper.data_filename)
        if i == 0:
            first_run_rec = r
        new_rec.before_learning_spikes.append(array(r.before_learning_nrn_spikes))
        new_rec.after_learning_spikes.append(array(r.after_learning_nrn_spikes))
        new_rec.realiz_target_nrn_spikes.append(array(r.realiz_target_nrn_spikes))
        new_rec.target_nrn_spikes.append(array(r.target_nrn_spikes))
    new_rec.src_filename = results_file    
    new_rec.saveInOneH5File(h5file)
    return new_rec, first_run_rec
    
def plot_spike_corr(corr, p):
    ep = p.experiment
    plot(corr, 'k-')
    plot(corr, 'kd', markersize = 5)
    xticks( arange(0,len(corr)) , [ "%d" % i for i in arange(0, int(ep.Tsim/60), int(ep.Tsim/60/6) ) ] )
    xlim(0,6.1)
    xlabel('time [min]') 
    ylabel('spike correlation')
    yticks( arange(0.50,0.95,0.1), [ '%.2f' % (x,) for x in arange(0.50,0.95,0.1) ] )
    ylim(0.50,0.91)
    
def plot_weightvec_angle(p, r):
    ep = p.experiment
    p = p.biofeed
    target_w = hstack((ones(p.numStrongTargetSynapses)*p.Wmax, zeros(p.numWeakTargetSynapses)*p.Wmax))
    norm_target_w = target_w / sqrt(inner(target_w , target_w))
    normed_weights = r.weights.copy()
    for i in range(normed_weights.shape[1]):
        normed_weights[:,i] /= sqrt(inner(normed_weights[:,i], normed_weights[:,i]))    
    angle = arccos(dot(norm_target_w, normed_weights))    
    plot(arange(0,len(angle)*ep.DTsim*p.samplingTime, ep.DTsim * p.samplingTime), angle, 'k-')
    xlim(0,ep.Tsim+1)        
    xticks(arange(0, ep.Tsim + 1, ep.Tsim/4.0), [ "%d" % i for i in arange(0, float(ep.Tsim+10)/60, int(ep.Tsim/60.0)/4.0 ) ] )
    xlabel('time [min]')
    ylabel('angular error [rad]')
    yticks(arange(0.0,1.01,0.2), [ "%.1f" % x for x in arange(0.0,1.01,0.2) ] )
    ylim(0.0,0.9)
    
def plot_weight_evolution(p, r, ax):
    ep=p.experiment
    
    box()
    xticks([])
    yticks([])
    
    ax_length = 0.8
    ax_gap = 0.08
    
    leg_width = 0.07
    
    ax_pos = ax.get_position().get_points().flatten()
    ax_pos[2] -= ax_pos[0]
    ax_pos[3] -= ax_pos[1]    
    
    leg_ax_pos = list(ax_pos)
    leg_ax_pos[0]  = leg_ax_pos[0] + leg_ax_pos[2]*(ax_length + ax_gap) 
    leg_ax_pos[2] = leg_width * leg_ax_pos[2]
    leg_ax = axes(leg_ax_pos)
    
    
    arr = arange(1,0,-0.01)
    arr.resize(100,1)    
    imshow(arr, aspect = 0.098)
    xticks([])
     
    yticks( arange(0,101,50), ['0', '0.5', '1'] )
    text(1.17, 0.5, '$w/w_{max}$', horizontalalignment = 'center', verticalalignment = 'center', rotation = 90, transform = ax.transAxes)
    
    leg_ax.yaxis.tick_right()
    
    
    new_ax_pos = list(ax_pos)
    new_ax_pos[2] = new_ax_pos[2] * ax_length   
    im_ax = axes(new_ax_pos)
    
    jet()
    imshow(r.weights, aspect = 1.7, interpolation = 'nearest')
    
    yticks( arange(0,101,50) , [ '%d' % (x,) for x in arange(100,-1,-50) ])
    xticks(arange(0, 201, 50), [ "%d" % i for i in arange(0, float(ep.Tsim+10)/60, int(ep.Tsim/60.0)/4.0 ) ] )
    xlabel('time [min]')        
    ylabel('synapse \#')
    jet()
    
    jet()    
    pass
    
    

def plot_multi_run_wstar(directory):    
    p = re.compile('biofeed.*\.h5')
    entries = os.listdir(directory)
    files = [ x for x in entries if p.match(x) ]    
    files.sort()
    print files
    plot_colors = [ 'b', 'r', 'g', 'm', 'k']
    col_n = 0
    for fname in files:
        h5file = openFile(os.path.join(directory,fname), mode = "r")

        all_p = constructParametersFromH5File(h5file)
        all_r = constructRecordingsFromH5File(h5file)
        
        h5file.close();
        
        p = all_p.biofeed
        ep = all_p.experiment
        
        r = all_r.biofeed
        
        strong_syn_avg = average(r.weights[:p.numStrongTargetSynapses], 0)
        strong_syn_std = std(r.weights[:p.numStrongTargetSynapses], 0)
        weak_syn_avg = average(r.weights[p.numStrongTargetSynapses:], 0)
        weak_syn_std = std(r.weights[p.numStrongTargetSynapses:], 0)

        plot( arange(0,(len(strong_syn_avg)-.5)*ep.DTsim*p.samplingTime, ep.DTsim * p.samplingTime), strong_syn_avg, plot_colors[col_n] + '-' )        
        plot( arange(0,(len(weak_syn_avg) -.5)*ep.DTsim*p.samplingTime, ep.DTsim * p.samplingTime), weak_syn_avg, plot_colors[col_n] + '--' )
        col_n += 1
        
    xlim(0,ep.Tsim+1)
    print "range is ", arange(0, ep.Tsim + 1, ep.Tsim/4.0)    
    xticks(arange(0, ep.Tsim + 1, ep.Tsim/4.0), [ "%d" % i for i in arange(0, float(ep.Tsim+10)/60, int(ep.Tsim/60.0)/4.0 ) ] )
    xlabel('time [min]')
    ylim(0,p.Wmax)
    yticks(arange(0,p.Wmax*1.001, p.Wmax/5.0), [ "%.1f" % i for i in arange(0,1.01,0.2) ])
    
    ylabel('avg. weights $(w/w_{max})$')
    
    
def plot_weight_change_fig(r, p):    
    last_weights = []
    initial_weights = []
    for w in r.weights:
        last_weights.append(mean(w[-10:-1]))
        initial_weights.append(w[0])
    
        failed_strong = len(find(last_weights[0:50] < p.Wmax/2))
        failed_weak = len(find(last_weights[50:100] > p.Wmax/2))
    
    plot(arange(100), hstack((p.Wmax * ones(50), 0 * zeros(50))), 'k:')
    plot(arange(100), p.Wmax/2 * ones(100), 'k--')
    plot(arange(100), initial_weights, 'k x', markersize = 3.4)
    plot(arange(100), last_weights, 'k o',markersize = 3.4)
    vlines(arange(100), initial_weights, last_weights)
    xlabel('synapse \#')
    ylim(0, p.Wmax)
    yticks(arange(0,p.Wmax*1.001, p.Wmax/5.0), [ "%.1f" % i for i in arange(0,1.01,0.2) ])
    ylabel('syn. weight $(w/w_{max})$')
    xticks( arange(0,101,50), [ '%d' % (x,) for x in arange(0,101,50)] )    
    
   
if __name__ == "__main__":
    mode = "just_corr"
    mode = 'complete'
    
    XBeforeMin, XBeforeMax = (5,35)   
    if mode == 'complete':
        if len(sys.argv) > 1:
            sim_dir = sys.argv[1]
        else:
            sim_dir = last_created_dir('biofeed.*')
        sim_file = os.path.join(sim_dir, last_file('biofeed.*er18.*', sim_dir))
        
        print " loading simulation filename : ", sim_file
        
        output_name = 'noname'
        if len(sys.argv) > 2:
            output_name = sys.argv[2]
        spikes_h5file = open_experiment_h5file("spikes_corr", output_name)
        
        new_rec, first_run_rec = multi_run_and_save_beforeAfter(sim_file, spikes_h5file, 6, XBeforeMax)        
    else:
        if len(sys.argv) > 1:
            sim_dir = sys.argv[1]
        else:
            sim_dir = last_created_dir('biofeed.*')
        sim_file = os.path.join(sim_dir, last_file('biofeed.*', sim_dir))            
        if len(sys.argv) > 1:
            spikes_h5file = sys.argv[1]
        else:
            spikes_h5file = last_file('spikes_corr.*\.h5$')        
        print " loading h5 filename : ", spikes_h5file
        sim_file = constructRecordingsFromH5File(spikes_h5file).src_filename

        print "loading sim h5 filename : " , sim_file
    
    
        
    sim_r = constructRecordingsFromH5File(sim_file).biofeed
    sim_p = constructParametersFromH5File(sim_file)
    
    pp = sim_p.biofeed
    
    A_plus_kappa_theory = pp.DAStdpRate * pp.stdpApos * pp.KappaApos / (0.01 * pp.Wmax)
    A_minus_kappa_theory = pp.DAStdpRate * pp.stdpApos * pp.KappaAneg / (0.01 * pp.Wmax)
    
    print "A_plus_kappa_theory = ", A_plus_kappa_theory
    print "A_minus_kappa_theory = ", A_minus_kappa_theory
    print " ratio of A_plus and A_minus kappas = ", A_plus_kappa_theory / A_minus_kappa_theory
    
        
    f = figure(1,figsize=(8,9), facecolor = 'w')
    
    f.subplots_adjust(top= 0.93, left = 0.11, bottom = 0.06, right = 0.93, hspace = 0.55, wspace = 0.55)
    clf()
    
    print sim_p
    
    ax = subplot(3, 2, 1, projection = 'frameaxes')
    text(-0.25, 1.13, 'A', fontsize = 'x-large', transform = ax.transAxes )
    plot_multi_run_wstar(sim_dir)
    
    
    
    ax = subplot(3, 2, 2, projection = 'frameaxes')
    text(-0.25, 1.13,  'B', fontsize = 'x-large', transform = ax.transAxes )
    start = 19.1 
    end = 21.6
    spike_idx = 0
    display_spike_trains(spikes_h5file, spike_idx, start, end, ax)

    
    ax = subplot(3, 2, 3, projection = 'frameaxes')
    text(-0.25, 1.13,  'C', fontsize = 'x-large', transform = ax.transAxes )
    corr = generate_spike_corr(spikes_h5file, XBeforeMin, XBeforeMax)
    plot_spike_corr(corr, sim_p)
    
    
    
    ax = subplot(3, 2, 4, projection = 'frameaxes')
    text(-0.25, 1.13,  'D', fontsize = 'x-large', transform = ax.transAxes )
    plot_weightvec_angle(sim_p, sim_r)
    
    
    
    ax = subplot(3, 2, 5)
    text(-0.25, 1.12,  'E', fontsize = 'x-large', transform = ax.transAxes )
    plot_weight_change_fig(sim_r, sim_p.biofeed)
    
    
    ax = subplot(3, 2, 6)
    text(-0.25, 1.12,  'F', fontsize = 'x-large', transform = ax.transAxes )    
    plot_weight_evolution(sim_p, sim_r, ax)    


    savefig("wstar_static_current.eps")
