#!/bin/bash
kernels=$1
dir=$2
metric=(gpu__time_duration.sum
        smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
        smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
        smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
        smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
        smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
        smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
        smsp__sass_thread_inst_executed_op_hadd_pred_on.sum
        smsp__sass_thread_inst_executed_op_hfma_pred_on.sum
        smsp__sass_thread_inst_executed_op_hmul_pred_on.sum
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        l1tex__t_bytes.sum                      #l1
        lts__t_bytes.sum                        #l2
        dram__bytes.sum                         #gmem
        sm__cycles_elapsed.avg.per_second
        sm__cycles_elapsed.avg
        sm__inst_executed_pipe_tensor.sum
        sm__warps_active.avg.pct_of_peak_sustained_active
        smsp__sass_average_branch_targets_threads_uniform.pct
        smsp__thread_inst_executed_per_inst_executed.ratio
        l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed
        smsp__cycles_active.avg.pct_of_peak_sustained_elapsed)

for kernel in ${kernels[@]}
do
	echo ${kernel}
	filename=${dir}/${kernel}/${kernel}
	echo ${filename}
	rm -f ${filename}.csv	
	
	for m in ${metric[@]}
	do
		echo $m
		data=$(grep -E "${kernel}.*${m}\"" ./${kernel}.log) 
		echo "${data}" >> ${filename}.csv
	done
done
