#!/bin/bash
set -x
#echo kernel
kernel=$1
exeval="$2"
metrics=""
nvprof --print-summary --csv $exeval |& tee gpu__time_duration.log
metric=(flop_count_sp
        flop_count_dp
        flop_count_hp
        inst_integer
        gld_transactions                                #l1 
        gst_transactions                                #l1 
        atomic_transactions                             #l1
        local_load_transactions                         #l1
        local_store_transactions                        #l1
        shared_load_transactions                        #l1
        shared_store_transactions                       #l1
        l2_read_transactions                            #l2 reads
        l2_write_transactions                           #l2 writes
        dram_read_transactions                          #gmem reads
        dram_write_transactions                         #gmem writes
        achieved_occupancy
        branch_efficiency
        warp_execution_efficiency
        shared_utilization
        sm_efficiency
        warp_nonpred_execution_efficiency)

for k in ${kernel[@]}
do
        echo "Profiling kernel: ${k}"
        for m in ${metric[@]}
        do
                metrics+=$m","
        done
        nvprof --kernels "${k}" --csv --metrics $metrics --csv $exeval |& tee ${k}_${m}.log
      
done
