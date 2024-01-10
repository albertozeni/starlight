import csv
import math
from re import T
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import os
import subprocess
import itertools
import json

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def extract_gpu_specs(kernel_name,profiling_results_name,json_file):
    location = profiling_results_name+kernel_name
    for filename in os.listdir(location):
        if filename == 'deviceQuery.log':
            f = open(os.path.join(location, 'deviceQuery.log'), "r")
            output = str(f.read())
            name = output.split('\\n')[0].split(
                'Device 0:')[1].split('"')[1].strip()
            gpu_clock = output.split('\\n')[0].split('Device 0:')[1].split(
                'GPU Max Clock rate:')[1].split('MHz')[0].strip()
            mem_clock = output.split('\\n')[0].split('Device 0:')[1].split(
                'Memory Clock rate:')[1].split('Mhz')[0].strip()
            bus_bit_width = output.split('\\n')[0].split('Device 0:')[1].split(
                'Memory Bus Width:')[1].split('-bit')[0].strip()
            cuda_cores = output.split('\\n')[0].split('Device 0:')[1].split(
                'CUDA Cores/MP:')[1].split('CUDA Cores')[0].strip()
            sm = int(cuda_cores)/int(output.split('\\n')[0].split('Device 0:')[1].split(
                'Multiprocessors, (')[1].split(') CUDA Cores/MP:')[0].strip())

            if (json_file is not None):
                with open(json_file) as f:
                    gpu_data = json.load(f)
                gpu_l1_bandwidth_val = gpu_data['L1_BW']
                gpu_l2_bandwidth_val = gpu_data['L2_BW']
                gpu_gm_bandwidth_val = gpu_data['GMEM_BW']
                gpu_fp_half_peak_val = gpu_data['HALF_PEAK']
                gpu_fp_double_peak_val = gpu_data['FP_DOUBLE_PEAK']
                gpu_fp_single_peak_val = gpu_data['FP_SINGLE_PEAK']
                gpu_int_peak_val = gpu_data['INT_PEAK']
            else:
                # gpu clock * #sm * 128 Bytes / 1000
                gpu_l1_bandwidth_val = int(gpu_clock)*int(sm)*128/(1E3)
                gpu_l2_bandwidth_val = int(gpu_clock)*int(sm)*32/(1E3)
                gpu_gm_bandwidth_val = int(mem_clock)*int(bus_bit_width)*2/(1E3*8)
                gpu_fp_half_peak_val = (int(cuda_cores)*int(gpu_clock)*2/1E3)*2
                gpu_fp_double_peak_val = (int(cuda_cores)*int(gpu_clock)*2/1E3)/32 #for rtx a5000 divide by 32, normal rtx 64, 100 series boards divide by 2
                gpu_fp_single_peak_val = int(cuda_cores)*int(gpu_clock)*2/1E3
                gpu_int_peak_val = gpu_fp_single_peak_val
            return name, gpu_l1_bandwidth_val, gpu_l2_bandwidth_val, gpu_gm_bandwidth_val, gpu_int_peak_val, gpu_fp_half_peak_val, gpu_fp_single_peak_val, gpu_fp_double_peak_val, cuda_cores


def setup_plot():
    """
    Standard setup of plot style;
    """
    # Reset matplotlib settings:
    plt.rcdefaults()
    # Set style:
    plt.rcParams["font.family"] = ["Serif"]
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = r"\usepackage{libertine}"
    plt.rcParams['axes.titlepad'] = 40
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.major.pad'] = 5
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def plot_performance_lines(peak_val, gpu_l1_bandwidth_val, gpu_l2_bandwidth_val, gpu_gm_bandwidth_val, oi_rightmost_point, type, colors, label_bandwidth):
    zero_l1 = [0.0000000001, 0.0000000001*gpu_l1_bandwidth_val]
    zero_l2 = [0.0000000001, 0.0000000001*gpu_l2_bandwidth_val]
    zero_gmem = [0.0000000001, 0.0000000001*gpu_gm_bandwidth_val]
    l1_ridge = [peak_val / gpu_l1_bandwidth_val, peak_val]
    l2_ridge = [peak_val / gpu_l2_bandwidth_val, peak_val]
    gmem_ridge = [peak_val / gpu_gm_bandwidth_val, peak_val]
    right_most = [oi_rightmost_point*2, peak_val]
    if(label_bandwidth == False):
        label_bandwidth = True
        x_values = [zero_l1[0], l1_ridge[0]]
        y_values = [zero_l1[1], l1_ridge[1]]
        plt.plot(x_values, y_values, colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 label="L1 CACHE BANDWIDTH: "+'%.2f' % gpu_l1_bandwidth_val+" GB/sec")
        x_values = [zero_l2[0], l2_ridge[0]]
        y_values = [zero_l2[1], l2_ridge[1]]
        plt.plot(x_values, y_values, colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 label="L2 CACHE BANDWIDTH: "+'%.2f' % gpu_l2_bandwidth_val+" GB/sec")
        x_values = [zero_gmem[0], gmem_ridge[0]]
        y_values = [zero_gmem[1], gmem_ridge[1]]
        plt.plot(x_values, y_values, colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 label="GMEM BANDWIDTH: "+'%.2f' % gpu_gm_bandwidth_val+" GB/sec")
    else:
        x_values = [zero_l1[0], l1_ridge[0]]
        y_values = [zero_l1[1], l1_ridge[1]]
        plt.plot(x_values, y_values, colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [zero_l2[0], l2_ridge[0]]
        y_values = [zero_l2[1], l2_ridge[1]]
        plt.plot(x_values, y_values, colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [zero_gmem[0], gmem_ridge[0]]
        y_values = [zero_gmem[1], gmem_ridge[1]]
        plt.plot(x_values, y_values, colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)

    if (type == 'DOUBLE'):
        peak_val_plot = "PEAK "+type + \
            " PERFORMANCE: "+'%.2f' % peak_val+" GFLOPs/sec"
        x_values = [l1_ridge[0], l2_ridge[0]]
        y_values = [l1_ridge[1], l2_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dotted')
        x_values = [l2_ridge[0], gmem_ridge[0]]
        y_values = [l2_ridge[1], gmem_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dotted')
        x_values = [gmem_ridge[0], right_most[0]]
        y_values = [gmem_ridge[1], right_most[1]]
        plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 color='black', linestyle='dotted', label=peak_val_plot)
        plt.plot(x_values, y_values,
                 colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dotted')

    if (type == 'INT'):
        peak_val_plot = "PEAK "+type+" PERFORMANCE: "+'%.2f' % peak_val+" GIOPs/sec"
        x_values = [l1_ridge[0], l2_ridge[0]]
        y_values = [l1_ridge[1], l2_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')
        x_values = [l2_ridge[0], gmem_ridge[0]]
        y_values = [l2_ridge[1], gmem_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')
        x_values = [gmem_ridge[0], right_most[0]]
        y_values = [gmem_ridge[1], right_most[1]]
        plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 color='black', linestyle='dashed', label=peak_val_plot)
        plt.plot(x_values, y_values,
                 colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')

    if (type == 'SINGLE'):
        peak_val_plot = "PEAK "+type + \
            " PERFORMANCE: "+'%.2f' % peak_val+" GFLOPs/sec"
        x_values = [l1_ridge[0], l2_ridge[0]]
        y_values = [l1_ridge[1], l2_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')
        # plt.plot(x_values, y_values,
        #          colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashdot')
        x_values = [l2_ridge[0], gmem_ridge[0]]
        y_values = [l2_ridge[1], gmem_ridge[1]]
        plt.plot(x_values, y_values,
                 colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')
        # plt.plot(x_values, y_values,
        #          colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashdot')
        x_values = [gmem_ridge[0], right_most[0]]
        y_values = [gmem_ridge[1], right_most[1]]
        plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 color='black', linestyle='dashed', label=peak_val_plot)
        plt.plot(x_values, y_values,
                 colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashed')
        # plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
        #          color='black', linestyle='dashdot', label=peak_val_plot)
        # plt.plot(x_values, y_values,
        #          colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1, linestyle='dashdot')

    if (type == 'HALF'):
        peak_val_plot = "PEAK "+type + \
            " PERFORMANCE: "+'%.2f' % peak_val+" GFLOPs/sec"
        x_values = [l1_ridge[0], l2_ridge[0]]
        y_values = [l1_ridge[1], l2_ridge[1]]
        plt.plot(x_values, y_values, colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [l2_ridge[0], gmem_ridge[0]]
        y_values = [l2_ridge[1], gmem_ridge[1]]
        plt.plot(x_values, y_values, colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [gmem_ridge[0], right_most[0]]
        y_values = [gmem_ridge[1], right_most[1]]
        plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 color='black', label=peak_val_plot)
        plt.plot(x_values, y_values, colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)

    if (type == 'TENSOR'):
        peak_val_plot = "PEAK "+type + \
            " PERFORMANCE: "+'%.2f' % peak_val+" GFLOPs/sec"
        x_values = [l1_ridge[0], l2_ridge[0]]
        y_values = [l1_ridge[1], l2_ridge[1]]
        plt.plot(x_values, y_values, colors[0], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [l2_ridge[0], gmem_ridge[0]]
        y_values = [l2_ridge[1], gmem_ridge[1]]
        plt.plot(x_values, y_values, colors[1], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)
        x_values = [gmem_ridge[0], right_most[0]]
        y_values = [gmem_ridge[1], right_most[1]]
        plt.plot(x_values, y_values, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1,
                 color='black', label=peak_val_plot)
        plt.plot(x_values, y_values, colors[2], path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], zorder=1)

    # handles.append(line_style)

    # ridge points
    plt.scatter(peak_val / gpu_l1_bandwidth_val, peak_val,
                s=60, color=colors[0], marker='o', edgecolors='black', zorder=2)
    plt.scatter(peak_val / gpu_l2_bandwidth_val, peak_val,
                s=60, color=colors[1], marker='o', edgecolors='black', zorder=2)
    plt.scatter(peak_val / gpu_gm_bandwidth_val, peak_val,
                s=60, color=colors[2], marker='o', edgecolors='black', zorder=2)
    return label_bandwidth
##############################
##############################


def plot_gpu(kernel_name, profiler, mode, version, json_file):
    setup_plot()
    # print(os.getenv('LANG'))
    label_bandwidth = False
    mode_flag = False
    profiling_results_name = "roofline_results/"
    if(mode==1 or mode==2):
        mode_flag = True
    if(mode==2):
        profiling_results_name = "optimization_results/"
    colors = ["#fc8d59", "#ffffbf", "#91bfdb"]
    kernel_folder_res = kernel_name
    styles = ['X', 'D', 'v', '^', '+', "*", "h", "H",
              "s", "1", "2", "3", "4", "8", "p", "d", "|", "_", ".", ","]
    # set which variable to plot
    plot_gpu = True
    emp_ceil = False
    print_zones = True
    # read GPU spec file
    GPUS_NAME, GPU_L1_BANDWIDTH, GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, GPU_INT_PEAK, GPU_FP_HALF_PEAK, GPU_FP_SINGLE_PEAK, GPU_FP_DOUBLE_PEAK, GPU_MULTIPROCESSORS = extract_gpu_specs(kernel_name,profiling_results_name, json_file)
    GPU_FP_TENSOR_PEAK = GPU_FP_DOUBLE_PEAK #to be fixed
    average_metric_index = 7
    # average_time_index = 15
    transactions_bytes_GMEM = 0
    transactions_bytes_l1cache = 0
    transactions_bytes_l2cache = 0

    flop_count_hp = 0
    flop_count_sp = 0
    flop_count_dp = 0
    int_count = 0
    tensor_ops = 0
    time = 0
    cc = 0
    cc_per_sec = 0
    achieved_occupancy = 0
    branch_efficiency = 0
    warp_execution_efficiency = 0
    warp_nonpred_exec_eff = 0
    shared_utilization = 0
    sm_efficiency = 0

    if(profiler == "NCU"):
        if(version < 2022400):
            average_metric_index = 11
        else:
            average_metric_index = 14
        list_len = 0
        # should be kernel name
        with open(os.path.join(profiling_results_name, kernel_folder_res, kernel_name+".csv"), 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                rowlist = list(row)
                # TIME WITH TIMER
                if("gpu__time_duration.sum") in rowlist:
                    list_len += 1
                    tmp = rowlist[average_metric_index]
                    tmp = tmp.replace(',', '')
                    tmp = tmp.replace('.', '')
                    tmp = int(tmp)
                    unit = rowlist[average_metric_index-1]
                    if("msecond" == unit):
                        tmp = tmp / 1E3
                    elif('usecond' == unit):
                        tmp = tmp / 1E6
                    elif('nsecond' == unit):
                        tmp = tmp / 1E9
                    elif('second' == unit):
                        tmp = tmp
                    else:
                        print("Unsupported time metric for %s" % (unit))
                        exit(-1)
                    time += tmp
                # ## TIME WITH CLOCK CYCLES
                # if("sm__cycles_elapsed.avg.per_second") in rowlist:
                #     list_len += 1
                #     # tmp = rowlist[average_metric_index]
                #     tmp = rowlist[average_metric_index].replace('.', '')
                #     tmp = tmp.replace(',','.')
                #     print(float(tmp))
                #     cc_per_sec += float(tmp)
                # if("sm__cycles_elapsed.avg") in rowlist:
                #     # tmp = rowlist[average_metric_index]
                #     tmp = rowlist[average_metric_index].replace('.', '')
                #     tmp = tmp.replace(',','.')
                #     cc += float(tmp)
                ## SINGLE FLOP COUNT
                if("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_sp += int(tmp)
                if("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_sp += int(tmp)*2
                if("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_sp += int(tmp)
                ## DOUBLE FLOP COUNT
                if("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_dp = int(tmp)
                if("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_dp += int(tmp)*2
                if("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_dp += int(tmp)
                ## HALF FLOP COUNT
                if("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_hp += int(tmp)
                if("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_hp += int(tmp)*2
                if("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    flop_count_hp += int(tmp)
                ## INT OPERATIONS COUNT
                if("smsp__sass_thread_inst_executed_op_integer_pred_on.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    int_count += int(tmp)
                ## L1 TRANSACTIONS
                if("l1tex__t_bytes.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    transactions_bytes_l1cache += int(tmp)
                ## L2 TRANSACTIONS
                if("lts__t_bytes.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    transactions_bytes_l2cache += int(tmp)
                ## DRAM TRANSACTIONS
                if("dram__bytes.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    transactions_bytes_GMEM += int(tmp)
                ## TENSOR HALF OPERATIONS
                if("sm__inst_executed_pipe_tensor.sum") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '') # if it is an integer we have to remove evenutal stuff dependent from the system language
                    tmp = tmp.replace('.', '')
                    tensor_ops += int(tmp)*512
                ## USEFUL PERFORMANCE METRICS
                if("sm__warps_active.avg.pct_of_peak_sustained_active") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    achieved_occupancy += float(tmp)
                if("smsp__sass_average_branch_targets_threads_uniform.pct") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    branch_efficiency += float(tmp)
                if("smsp__thread_inst_executed_per_inst_executed.ratio") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    warp_execution_efficiency += float(tmp)
                if("l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    shared_utilization += float(tmp)
                if("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed") in rowlist:
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    sm_efficiency += float(tmp)
                if("smsp__thread_inst_executed_per_inst_executed.pct") in rowlist: #to be checked
                    tmp = rowlist[average_metric_index].replace(',', '.')
                    warp_nonpred_exec_eff += float(tmp)
                

        if(list_len==0):
            print(color.RED + color.BOLD + "Nothing has been profiled, check if the application exists" + color.END) 
            exit() 
        print(color.BOLD+"Generating roofline"+color.END)
        flop_count_dp /= list_len
        flop_count_sp /= list_len
        flop_count_hp /= list_len
        tensor_ops /= list_len
        int_count /= list_len
        transactions_bytes_l1cache /= list_len
        transactions_bytes_l2cache /= list_len
        transactions_bytes_GMEM /= list_len
        achieved_occupancy /= list_len
        branch_efficiency /= list_len
        warp_execution_efficiency /= list_len
        warp_nonpred_exec_eff /= list_len
        shared_utilization /= list_len
        sm_efficiency /= list_len
        # cc /= list_len
        # cc_per_sec /= list_len
        # time = cc/cc_per_sec 
        time /= list_len

    elif(profiler == "NVPROF"):
        average_metric_index = 7
        # should be kernel name
        with open(os.path.join(profiling_results_name, kernel_folder_res, kernel_name+".csv"), 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                rowlist = list(row)
                # if("0") in rowlist:
                if("GPU activities") in rowlist:
                    time = float(rowlist[4])
                    # time = time.replace(',', '')
                    # time = time.replace('.', '')
                    # time = int(time)
                    unit = "msecond"
                    if("msecond" == unit):
                        time = time / 1E3
                    elif('usecond' == unit):
                        time = time / 1E6
                    elif('nsecond' == unit):
                        time = time / 1E9
                    elif('second' == unit):
                        time = time
                    else:
                        print("Unsupported time metric for %s" % (unit))
                        exit(-1)
                ## DOUBLE FLOP COUNT
                if("flop_count_dp") in rowlist:
                    tmp = rowlist[average_metric_index]
                    flop_count_dp = int(tmp)
                ## SINGLE FLOP COUNT 
                if("flop_count_sp") in rowlist:
                    tmp = rowlist[average_metric_index]
                    flop_count_sp = int(tmp)
                ## HALF FLOP COUNT
                if("flop_count_hp") in rowlist:
                    tmp = rowlist[average_metric_index]
                    flop_count_hp = int(tmp)
                ## INT OP/INST COUNT
                if("inst_integer") in rowlist:
                    tmp = rowlist[average_metric_index]
                    int_count += int(tmp)
                ## L1 TRANSACTIONS
                if("gld_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("gst_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("atomic_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("local_load_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("local_store_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("shared_load_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                if("shared_store_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l1cache += int(tmp)
                ## L2 TRANSACTIONS
                if("l2_read_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l2cache += int(tmp)
                if("l2_write_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_l2cache += int(tmp)
                ## DRAM TRANSACTIONS
                if("dram_read_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_GMEM += int(tmp)
                if("dram_write_transactions") in rowlist:
                    tmp = rowlist[average_metric_index]
                    transactions_bytes_GMEM += int(tmp)
                ## USEFUL PERF METRICS    
                if("achieved_occupancy") in rowlist:
                    achieved_occupancy += float(tmp)
                if("branch_efficiency") in rowlist:
                    branch_efficiency += float(tmp)
                if("warp_execution_efficiency") in rowlist:
                    warp_execution_efficiency += float(tmp)
                if("shared_utilization") in rowlist:
                    shared_utilization += float(tmp)
                if("sm_efficiency") in rowlist:
                    sm_efficiency += float(tmp)
                if("warp_nonpred_execution_efficiency") in rowlist:
                    warp_nonpred_exec_eff += float(tmp)          
    
    int_p = False
    half_p = False
    single_p = False
    double_p = False
    tensor_p = False

    performance_double = sys.float_info.max
    performance_single = sys.float_info.max
    performance_int = sys.float_info.max
    performance_half = sys.float_info.max
    performance_tensor = sys.float_info.max

    l1_cache_x_int = sys.float_info.max
    l2_cache_x_int = sys.float_info.max
    gmem_x_int = sys.float_info.max
    l1_cache_x_half = sys.float_info.max
    l2_cache_x_half = sys.float_info.max
    gmem_x_half = sys.float_info.max
    l1_cache_x_single = sys.float_info.max
    l2_cache_x_single = sys.float_info.max
    gmem_x_single = sys.float_info.max
    l1_cache_x_double = sys.float_info.max
    l2_cache_x_double = sys.float_info.max
    gmem_x_double = sys.float_info.max
    l1_cache_x_tensor = sys.float_info.max
    l2_cache_x_tensor = sys.float_info.max
    gmem_x_tensor = sys.float_info.max

    if(profiler=="NVPROF"):
        transactions_bytes_l1cache*=32
        transactions_bytes_l2cache*=32
        transactions_bytes_GMEM*=32

    if(int_count > 0):
        l1_cache_x_int = (int_count)/(transactions_bytes_l1cache)
        l2_cache_x_int = (int_count)/(transactions_bytes_l2cache)
        gmem_x_int = (int_count)/(transactions_bytes_GMEM)
        performance_int = (int_count/(time))/1E9
        # print(performance_int)
        if(mode_flag):
            int_p = True
        elif (performance_int>=100):
            int_p = True

    if(flop_count_hp > 0):
        l1_cache_x_half = (flop_count_hp)/(transactions_bytes_l1cache)
        l2_cache_x_half = (flop_count_hp)/(transactions_bytes_l2cache)
        gmem_x_half = (flop_count_hp)/(transactions_bytes_GMEM)
        performance_half = (flop_count_hp/(time))/1E9
        if(mode_flag):
            half_p = True
        elif (performance_half>=100):
            half_p = True

    if(flop_count_sp > 0):
        l1_cache_x_single = (flop_count_sp)/(transactions_bytes_l1cache)
        l2_cache_x_single = (flop_count_sp)/(transactions_bytes_l2cache)
        gmem_x_single = (flop_count_sp)/(transactions_bytes_GMEM)
        performance_single = (flop_count_sp/(time))/1E9
        if(mode_flag):
            single_p = True
        elif (performance_single>=100):
            single_p = True

    if(flop_count_dp > 0):
        l1_cache_x_double = (flop_count_dp)/(transactions_bytes_l1cache)
        l2_cache_x_double = (flop_count_dp)/(transactions_bytes_l2cache)
        gmem_x_double = (flop_count_dp)/(transactions_bytes_GMEM)
        performance_double = (flop_count_dp/(time))/1E9
        if(mode_flag):
            double_p = True
        elif (performance_double>=100):
            double_p = True
    
    if(tensor_ops > 0):
        l1_cache_x_tensor = (tensor_ops)/(transactions_bytes_l1cache)
        l2_cache_x_tensor = (tensor_ops)/(transactions_bytes_l2cache)
        gmem_x_tensor = (tensor_ops)/(transactions_bytes_GMEM)
        performance_tensor = (tensor_ops/(time))/1E9
        if(mode_flag):
            tensor_p = True
        elif (performance_tensor>=100):
            tensor_p = True

    if(tensor_p == False and double_p == False and single_p == False and half_p == False and int_p == False):
            print(color.RED + color.BOLD + "The profiled application has very low, or none GPU operation" + color.END) 
            exit()

    #compute max perf
    max_performance = 0
    max_oi = 0
    peak_performance = 0
    #compute corresponding operational intensity
    if(performance_single < sys.float_info.max and performance_single >= max_performance):
        max_performance = performance_single
        max_oi = gmem_x_single
        peak_performance = GPU_FP_SINGLE_PEAK
    if(performance_int < sys.float_info.max and performance_int >= max_performance):
        max_performance = performance_int
        max_oi = gmem_x_int
        peak_performance = GPU_INT_PEAK
    if(performance_double < sys.float_info.max and performance_double >= max_performance):
        max_performance = performance_double
        max_oi = gmem_x_double
        peak_performance = GPU_FP_DOUBLE_PEAK
    if(performance_half < sys.float_info.max and performance_half >= max_performance):
        max_performance = performance_half
        max_oi = gmem_x_half
        peak_performance = GPU_FP_HALF_PEAK
    if(performance_tensor < sys.float_info.max and performance_tensor >= max_performance):
        max_performance = performance_tensor
        max_oi = gmem_x_tensor
        peak_performance = GPU_FP_TENSOR_PEAK

    min_performance = min(performance_single,
                          performance_int, performance_half,performance_double,performance_tensor)

    min_intensity = min(GPU_FP_DOUBLE_PEAK/GPU_GM_BANDWIDTH,l1_cache_x_int, l2_cache_x_int, gmem_x_int, l1_cache_x_half, l2_cache_x_half, gmem_x_half,
                        l1_cache_x_single, l2_cache_x_single, gmem_x_single, l1_cache_x_double, l2_cache_x_double, gmem_x_double, l1_cache_x_tensor, l2_cache_x_tensor, gmem_x_tensor)

    oi_rightmost_point = 0

    if(double_p):
        label_perf = "DOUBLE PERFORMANCE: " + \
            '%.2f' % performance_double+" GFLOPs/sec"
        if(performance_double < 0.01):
            label_perf = "DOUBLE PERFORMANCE below 0.01 GFLOPs/sec"
        plt.scatter(gmem_x_double, performance_double, s=100, color='white',
                    marker=styles[0], zorder=3, edgecolors='black', label=label_perf)
        plt.scatter(l1_cache_x_double, performance_double, s=100, color=colors[0],
                    marker=styles[0], zorder=3, edgecolors='black')
        plt.scatter(l2_cache_x_double, performance_double, s=100, color=colors[1],
                    marker=styles[0], zorder=3, edgecolors='black')
        plt.scatter(gmem_x_double, performance_double, s=100, color=colors[2],
                    marker=styles[0], zorder=3, edgecolors='black')
        oi_rightmost_point = max(
            oi_rightmost_point, l1_cache_x_double, l2_cache_x_double, gmem_x_double)
        # handles.append(point_style)

    if(int_p):
        label_perf = "INT PERFORMANCE: " + \
            '%.2f' % performance_int+" GIOPs/sec"
        if(performance_int < 0.01):
            label_perf = "INT PERFORMANCE below 0.01 GIOPs/sec"
        plt.scatter(gmem_x_int, performance_int, s=100, color='white',
                    marker=styles[1], zorder=3, edgecolors='black', label=label_perf)
        plt.scatter(l1_cache_x_int, performance_int, s=100, color=colors[0],
                    marker=styles[1], zorder=3, edgecolors='black')
        plt.scatter(l2_cache_x_int, performance_int, s=100, color=colors[1],
                    marker=styles[1], zorder=3, edgecolors='black')
        plt.scatter(gmem_x_int, performance_int, s=100, color=colors[2],
                    marker=styles[1], zorder=3, edgecolors='black')
        oi_rightmost_point = max(
            oi_rightmost_point, l1_cache_x_int, l2_cache_x_int, gmem_x_int)
        # handles.append(point_style)

    if(single_p):
        label_perf = "SINGLE PERFORMANCE: " + \
            '%.2f' % performance_single+" GFLOPs/sec"
        if(performance_single < 0.01):
            label_perf = "SINGLE PERFORMANCE below 0.01 GFLOPs/sec"
        plt.scatter(gmem_x_single, performance_single, s=100, color='white',
                    marker=styles[2], zorder=3, edgecolors='black', label=label_perf)
        plt.scatter(l1_cache_x_single, performance_single, s=100, color=colors[0],
                    marker=styles[2], zorder=3, edgecolors='black')
        plt.scatter(l2_cache_x_single, performance_single, s=100, color=colors[1],
                    marker=styles[2], zorder=3, edgecolors='black')
        plt.scatter(gmem_x_single, performance_single, s=100, color=colors[2],
                    marker=styles[2], zorder=3, edgecolors='black')
        oi_rightmost_point = max(
            oi_rightmost_point, l1_cache_x_single, l2_cache_x_single, gmem_x_single)
        # handles.append(point_style)

    if(half_p):
        label_perf = "HALF PERFORMANCE: " + \
            '%.2f' % performance_half+" GFLOPs/sec"
        if(performance_half < 0.01):
            label_perf = "HALF PERFORMANCE below 0.01 GFLOPs/sec"
        plt.scatter(gmem_x_half, performance_half, s=100, color='white',
                    marker=styles[3], zorder=3, edgecolors='black', label=label_perf)
        plt.scatter(l1_cache_x_half, performance_half, s=100, color=colors[0],
                    marker=styles[3], zorder=3, edgecolors='black')
        plt.scatter(l2_cache_x_half, performance_half, s=100, color=colors[1],
                    marker=styles[3], zorder=3, edgecolors='black')
        plt.scatter(gmem_x_half, performance_half, s=100, color=colors[2],
                    marker=styles[3], zorder=3, edgecolors='black')
        oi_rightmost_point = max(
            oi_rightmost_point, l1_cache_x_half, l2_cache_x_half, gmem_x_half)

    if(tensor_p):
        label_perf = "TENSOR PERFORMANCE: " + \
            '%.2f' % performance_tensor +" GFLOPs/sec"
        if(performance_tensor < 0.01):
            label_perf = "TENSOR PERFORMANCE below 0.01 GFLOPs/sec"
        plt.scatter(gmem_x_tensor, performance_tensor, s=100, color='white',
                    marker=styles[4], zorder=3, edgecolors='black', label=label_perf)
        plt.scatter(l1_cache_x_tensor, performance_tensor, s=100, color=colors[0],
                    marker=styles[4], zorder=3, edgecolors='black')
        plt.scatter(l2_cache_x_tensor, performance_tensor, s=100, color=colors[1],
                    marker=styles[4], zorder=3, edgecolors='black')
        plt.scatter(gmem_x_tensor, performance_tensor, s=100, color=colors[2],
                    marker=styles[4], zorder=3, edgecolors='black')
        oi_rightmost_point = max(
            oi_rightmost_point, l1_cache_x_tensor, l2_cache_x_tensor, gmem_x_tensor)

    peak_plot = 0

    if(plot_gpu):
        if(tensor_p):
            label_bandwidth=plot_performance_lines(GPU_FP_TENSOR_PEAK, GPU_L1_BANDWIDTH,
                                   GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, oi_rightmost_point, "TENSOR", colors, label_bandwidth)
            peak_plot = GPU_FP_TENSOR_PEAK
        
        if(double_p):
            label_bandwidth=plot_performance_lines(GPU_FP_DOUBLE_PEAK, GPU_L1_BANDWIDTH,
                                   GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, oi_rightmost_point, "DOUBLE", colors, label_bandwidth)
            peak_plot = GPU_FP_DOUBLE_PEAK

        if(int_p):
            label_bandwidth=plot_performance_lines(GPU_INT_PEAK, GPU_L1_BANDWIDTH,
                                   GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, oi_rightmost_point, "INT", colors, label_bandwidth)
            peak_plot = GPU_INT_PEAK

        if(single_p):
            label_bandwidth=plot_performance_lines(GPU_FP_SINGLE_PEAK, GPU_L1_BANDWIDTH,
                                   GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, oi_rightmost_point, "SINGLE", colors, label_bandwidth)
            peak_plot = GPU_FP_SINGLE_PEAK

        if(half_p):
            label_bandwidth=plot_performance_lines(GPU_FP_HALF_PEAK, GPU_L1_BANDWIDTH,
                                   GPU_L2_BANDWIDTH, GPU_GM_BANDWIDTH, oi_rightmost_point, "HALF", colors, label_bandwidth)
            peak_plot = GPU_FP_HALF_PEAK

    if(double_p or single_p or half_p or tensor_p):
        x_axis = "FLOPs/B"
        y_axis = "GFLOPs/sec"
        if single_p:
            x_axis = "FLOPs/B - IOPs/B"
            y_axis = "GFLOPs/sec - GIOPs/sec"
    else:
        x_axis = "IOPs/B"
        y_axis = "GIOPs/sec"

    # set logaritmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Plot grid Roofline
    plt.grid(True, axis='y', color='#bdbdbd',
             linestyle='-', linewidth=0.5, zorder=0)

    # Plot Axis Labels
    plt.xlabel("Operational Intensity [" + x_axis + "]")
    plt.ylabel("Performance [" + y_axis + "]")
    # plt.xlim([(peak_plot/GPU_L1_BANDWIDTH)/10, oi_rightmost_point*2])
    plt.ylim([min_performance/10, peak_plot*2])
    plt.xlim([min_performance/10/GPU_L1_BANDWIDTH, oi_rightmost_point*2])
    plt.axvline(peak_plot/GPU_GM_BANDWIDTH,
                linestyle=':', color='#bdbdbd', zorder=0)

    zero_l1 = [0.0000000001, 0.0000000001*GPU_L1_BANDWIDTH]
    l1_ridge = [peak_plot/GPU_L1_BANDWIDTH, peak_plot]
    gmem_ridge = [peak_plot/GPU_GM_BANDWIDTH, peak_plot]
    zero_gmem_ridge = [peak_plot/GPU_GM_BANDWIDTH, 0.0000000001]
    if(print_zones):
        # Fill in area underneath memory-bandwidth line
        coord = [zero_l1, l1_ridge, gmem_ridge, zero_gmem_ridge, zero_l1]
        xs, ys = zip(*coord)
        plt.fill(xs, ys, colors[0], alpha=.2,
                 zorder=0, label='MEMORY BOUND AREA')
        # FIll in area underneath compute line
        right_most = [oi_rightmost_point*2, peak_plot]
        right_most_zero = [oi_rightmost_point*2, 0.0000000001]
        coord = [zero_gmem_ridge, gmem_ridge,
                 right_most, right_most_zero, zero_gmem_ridge]
        xs, ys = zip(*coord)
        plt.fill(xs, ys, colors[2], alpha=.2,
                 zorder=0, label='COMPUTE BOUND AREA')
    # Plot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    cols = 1
    # plt.legend(flip(handles, cols), flip(labels, cols), numpoints=1, loc='center right', bbox_to_anchor=(1.02, 0., 1, 1),
    #            ncol=cols, mode="expand", borderaxespad=0)
    plt.legend(flip(handles, cols), flip(labels, cols), numpoints=1, loc='lower left',
               ncol=cols, bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0)
    # Save picture
    # plt.savefig(os.path.join(profiling_results_name, kernel_name,
    #             "roofline-"+kernel_name + ".png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(profiling_results_name, kernel_name,
                "roofline-"+kernel_name + ".pdf"), bbox_inches='tight')
    # Return performance info from roofline
    return peak_performance, max_performance, max_oi, GPU_GM_BANDWIDTH, gmem_ridge[0], achieved_occupancy, branch_efficiency, warp_nonpred_exec_eff, shared_utilization, sm_efficiency
