#!/bin/bash
kernels=$1
dir=$2
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
		
for kernel in ${kernels[@]}
do
	echo ${kernel}
	filename=${dir}/${kernel}/${kernel}
	echo ${filename}
	rm -f ${filename}.csv	
	
	for m in ${metric[@]}
	do
		echo $m
		data=$(grep -rin -E "${kernel}.*${m}" ./${kernel}_*.log) 
		data=$(echo "${data}"| cut -d : -f 2-)
		echo "${data}" >> ${filename}.csv
	done
	echo time
	data=`grep -rin -E "GPU activities" ./gpu__time_duration.log | cut -d : -f 2-`
	echo "${data}" >> gpu_activities.log
	data=`grep -rin -E $kernel ./gpu__time_duration.log | cut -d : -f 2-`
	echo "${data}" >> ${filename}.csv
done
