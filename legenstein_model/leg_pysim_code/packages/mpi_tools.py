from datetime import datetime
import os


def append_attr(line, attrname, value):
    line += " -" + attrname + " " + str(value) + " "
    return line 

def cluster_mpi_execute(executable,args, machine_list, attributes = {}, stdout = None):
    filename = 'mpi_config_' + "_".join([executable] + args) + "_" + datetime.today().strftime("%Y%m%d_%H%M%S") + ".mpi" 
    f = open(filename, 'w')
    f.write(' -l ')    
    for machine, np in machine_list.iteritems():
        line = ''
        line = append_attr(line, "n", np)
        line = append_attr(line, "host", machine)
        if machine in attributes:
            for k,v in attributes[machine].iteritems():
                line = append_attr(line,k,v)        
        line += " ".join([executable] + args)  + "\n"
        f.write(line)
    f.close()
    if stdout is None:
        stdout = 'stdout_' + "_".join([executable] + args) + datetime.strftime("%Y%m%d_%H%M%S") + ".out"
    cmd = 'nohup mpirun -configfile ' + filename + ' < /dev/null ' + ' &> ' + stdout + ' & '
    os.system(cmd)
        