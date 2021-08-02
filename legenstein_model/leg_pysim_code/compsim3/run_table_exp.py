import sys
import biofeedback_poiss_dbl_exp
reload(biofeedback_poiss_dbl_exp)

directory = sys.argv[2]

print "directory is ", directory

params = [[ "table_dbl_exp_1" , {"synTau":10e-3, "wScale":1.2, "Rbase":10, "stdpA":0.0005 * 2.77, "alpha":1.05, "stdpTaupos":20e-3,
           "stdpTauneg" : 20e-3, "KappaTaupos" : 50e-3 / 2.5, "KappaTauneg" : 50e-3 / 2.5, "KappaAlpha":1.07, 
           "NumSyn" : 100, "inputRate" : 6, "Tsim": 5.0 * 3600, 'directory':directory } ], 
           [ "table_dbl_exp_2" , {"synTau":7e-3, "wScale":2.0, "Rbase":5.0, "stdpA":0.0002 * 2.77, "alpha":1.02, "stdpTaupos":15e-3, 
           "stdpTauneg" : 15e-3, "KappaTaupos" : 40e-3 / 2.5, "KappaTauneg" : 40e-3 / 2.5, "KappaAlpha":1.10, 
           "NumSyn" : 100, "inputRate" : 6, "Tsim": 10.0 * 3600, 'directory':directory } ],
           [ "table_dbl_exp_3" , {"synTau":20e-3, "wScale":1.0, "Rbase":6.0, "stdpA":0.0002 * 2.77, "alpha":1.10, "stdpTaupos":25e-3, 
           "stdpTauneg" : 25e-3, "KappaTaupos" : 100e-3 / 2.5, "KappaTauneg" : 100e-3 / 2.5, "KappaAlpha":1.08,
           "NumSyn" : 100, "inputRate" : 6, "Tsim": 19.9 * 3600, 'directory':directory } ],
           [ "table_dbl_exp_4" , {"synTau":7e-3, "wScale":2.0, "Rbase":5.0, "stdpA":0.0002 * 2.77, "alpha":1.07, "stdpTaupos":25e-3, 
           "stdpTauneg" : 25e-3, "KappaTaupos" : 40e-3 / 2.5, "KappaTauneg" : 40e-3 / 2.5, "KappaAlpha":1.12,
           "NumSyn" : 100, "inputRate" : 6, "Tsim": 13.0 * 3600, 'directory':directory } ],           
           [ "table_dbl_exp_5" , {"synTau":10e-3, "wScale":1.5, "Rbase":6.0, "stdpA":0.0005 * 2.77, "alpha":1.10, "stdpTaupos":25e-3, 
           "stdpTauneg" : 25e-3, "KappaTaupos" : 50e-3 / 2.5, "KappaTauneg" : 50e-3 / 2.5, "KappaAlpha":1.20,
           "NumSyn" : 100, "inputRate" : 6, "Tsim": 2.0 * 3600, 'directory':directory } ],
           [ "table_dbl_exp_6" , {"synTau":25e-3, "wScale":1.0, "Rbase":3.0, "stdpA":0.0010 * 2.77, "alpha":1.01, "stdpTaupos":25e-3, 
           "stdpTauneg" : 25e-3, "KappaTaupos" : 50e-3 / 2.5, "KappaTauneg" : 50e-3 / 2.5, "KappaAlpha":1.07,
           "NumSyn" : 200, "inputRate" : 3, "Tsim" : 18.0 * 3600, 'directory':directory } ]
           ]

biofeedback_poiss_dbl_exp.experiment(params[int(sys.argv[1])][1],params[int(sys.argv[1])][0])





