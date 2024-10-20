universe = vanilla
getenv = True

initialdir = /home/divya.rana/github/gw_cogwheel_runs/output/

accounting_group =ligo.sim.o4.cbc.pe.bilby 
# Replace here your user name
accounting_group_user = divya.rana
# Replace here the folder you are running from
executable = /home/divya.rana/github/gw_cogwheel_runs/run_PE.sh
arguments = $(rel_no) $(run_no)

request_cpus   = 1
#request_memory = 1024M
#request_disk   = 100M

request_disk=5000M
request_memory=5000M

log =    log_PE/std_$(rel_no)_$(run_no).log
output = log_PE/std_$(rel_no)_$(run_no).out
error =  log_PE/std_$(rel_no)_$(run_no).err

transfer_output_files = output/
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
#when_to_transfer_output = ON_EXIT

#queue 
queue rel_no, run_no from resubmit.dat
