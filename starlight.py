#!/usr/bin/python3

import os, glob
from statistics import mode
import sys
import io
import subprocess

import tool_scripts.plot_ascii_art as paa
import tool_scripts.optimization_suggestions as ops

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   ORANGE = '\033[38;5;208m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def list_to_path(l): 
    str = ""  
    for w in l: 
        str += w + "/"    
    return str

def kernel_sample(kernel_name, app_path, to_profile):
    print(color.BOLD+"Sampling application"+color.END)
    os.system('./libpc_sampling_continuous.pl --app "%s" > /dev/null 2> /dev/null'%to_profile)
    os.system('cp 1_pcsampling.dat optimization_results/%s'%kernel_name)
    ### Cubin generation and renaming
    os.system('cuobjdump -xelf all %s > /dev/null 2> /dev/null'%app_path)
    cwd = os.getcwd()
    files = glob.glob("*.cubin")
    for index, file in enumerate(files):
        os.rename(os.path.join(cwd, file), os.path.join(cwd, ''.join([str(index+1), '.cubin'])))
    os.system('cp *.cubin optimization_results/%s'%kernel_name)
    ### Extraction
    os.system('./pc_sampling_utility --file-name 1_pcsampling.dat > sample_raw.csv 2> /dev/null')# check the err message for cubins    
    os.system('cp sample_raw.csv optimization_results/%s'%kernel_name)
    os.system('rm -f libpc_sampling_continuous.pl pc_sampling_utility 1_pcsampling.dat *.cubin sample_raw.csv')
    print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Sampling has been performed, the raw report has been generated in "+ color.GREEN+ cwd +"/optimization_results/"+kernel_name+color.END)

def build_sampler():
    cuda_v = os.popen('nvcc -V | grep release').read()
    # print(color.BOLD+"Building the sampling software with the current nvcc ("+ cuda_v +")")
    cwd = os.getcwd()
    os.chdir('tool_scripts/pc_sampling/')
    os.system('make > /dev/null 2> /dev/null')
    sampling_dir = os.getcwd()
    path_nvcc_lib = os.popen("which nvcc | sed 's/.........$//' | tr -d $'\n'").read()+"/lib64"
    os.system('cp libpc_sampling_continuous.pl ../../')
    os.system('cp pc_sampling_utility ../../')
    os.chdir(cwd)
    os.environ['LD_LIBRARY_PATH'] = os.pathsep + sampling_dir + os.pathsep + path_nvcc_lib
    
def parse_makefile(makefile_path):
    file_exists = os.path.exists(makefile_path)
    if(file_exists):
        makefile_path_list = makefile_path.split("/")
        # print(makefile_path_list)
        makefile_path_list_size = len(makefile_path_list)
        makefile_name = makefile_path_list[makefile_path_list_size-1]
        makefile_path_list.pop()
        path_to_makefile = list_to_path(makefile_path_list)
        # print(path_to_makefile)
        with open(makefile_path,'r') as file:
            filedata = file.read()
        filedata = filedata.replace('nvcc ', 'nvcc -lineinfo ')
        filedata = filedata.replace('gcc ', 'gcc -lineinfo ')
        filedata = filedata.replace('cc ', 'cc -lineinfo ')
        filedata = filedata.replace('clang ', 'clang -lineinfo ')
        filedata = filedata.replace('g++ ', 'g++ -lineinfo ')
        filedata = filedata.replace('clang++ ', 'clang++ -lineinfo ')
        # os.system("cp %s %s"%(makefile_path,path_to_makefile+"Makefile_modded"))ù
        with open(path_to_makefile+"Makefile_modded", 'w') as file:
            file.write(filedata)
        print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Makefile successfully parsed, created new makefile: "+color.PURPLE+path_to_makefile+"Makefile_modded"+color.END)
    else:
        print(color.BOLD+color.RED+"Makefile ("+makefile_path+") does not exist"+color.END)
        
def run_roofline(kernel_name, to_profile, path_to_exe, mode, device_id, json_file):
    
    if (json_file is not None):
        file_exists = os.path.exists(json_file)
        if(file_exists==False):
            print(color.BOLD+color.RED+"JSON file ("+json_file+") does not exist"+color.END)
            exit()
    file_exists = os.path.exists(path_to_exe)
    if(file_exists==False):
        print(color.BOLD+color.RED+"Exefile ("+path_to_exe+") does not exist"+color.END)
        exit()
    if(mode==0 or mode==1):
        profiling_results_name = "roofline_results/"
    elif(mode==2):
        profiling_results_name = "optimization_results/"

    os.system('rm -rf %s'%profiling_results_name+kernel_name)
    os.system('mkdir -p %s'%profiling_results_name+kernel_name)

    print(color.BOLD+"Profiling algorithm: "+kernel_name+color.END)
    #COMPILE GPU DEVICE QUERY AND EXTRACT GPU COMPUTE CAPABILITY
    print(color.BOLD+"Detecting GPU"+str(device_id)+" Compute Capability"+color.END)
    cwd = os.getcwd()
    os.chdir('tool_scripts/')
    os.system('git clone https://github.com/nvidia/cuda-samples.git > /dev/null 2> /dev/null')
    os.chdir('cuda-samples/Samples/1_Utilities/deviceQuery')
    os.system('make > /dev/null 2> /dev/null')
    os.chdir(cwd)
    os.system('rm -rf bin/')
    os.system('rm -rf common/')
    os.system('./tool_scripts/cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery > %s' %profiling_results_name+kernel_name+"/deviceQuery_tot.log")
    #for future developments, change Device 0 and Device 1 to desired Device n and Device n+1, it works even if you have only 1 gpu
    os.system("sed -n -e '/Device %s/,/Device %s/ p' %s > %s"%(str(device_id),str(device_id+1),profiling_results_name+kernel_name+"/deviceQuery_tot.log",profiling_results_name+kernel_name+"/deviceQuery.log"))
    string = profiling_results_name+kernel_name
    compute_capability = float(os.popen('echo `grep -rin -E "CUDA Capability Major/Minor version number:" %s/deviceQuery.log | cut -d : -f 3-`' %string).read())
    # print(compute_capability)
    #CHECK COMPUTE CAPABILITY TO USE THE CORRECT BENCHMARKING TOOL
    if (compute_capability<7):
        print(color.BOLD+"Compute capability " + str(compute_capability) +", metrics will be collected using nvprof"+color.END)
        print(color.BOLD+"Collecting GPU metrics"+color.END)
        os.system('bash ./tool_scripts/run_nvprof.sh %s "%s" > %s/profile_%s.log 2> %s/profile_%s_stderr.log'%(kernel_name,to_profile,profiling_results_name+kernel_name,kernel_name,profiling_results_name+kernel_name,kernel_name))
        print(color.BOLD+"Aggregating collected metrics"+color.END)
        os.system('bash ./tool_scripts/merge_nvprof.sh %s %s > /dev/null' %(kernel_name,profiling_results_name))
        os.system('rm -f *.log')
        PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY = roofline.plot_gpu(kernel_name,"NVPROF",mode,0,json_file)
        print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Roofline has been generated, the results are in "+ cwd +"/"+profiling_results_name+kernel_name+color.END)
    else:
        version = 0
        os.system("ncu -v | grep Version | cut -d ' ' -f 2 > ver.log")
        with open('ver.log', 'r') as f: #open the file
            for line in f:
                version = line
        version = version.replace('.','')
        version = int(version)
        print(color.BOLD+"Compute capability " + str(compute_capability) +", metrics will be collected using ncu"+color.END)
        print(color.BOLD+"Collecting GPU metrics"+color.END)
        os.system('bash ./tool_scripts/run_ncu.sh %s "%s" > %s/profile_%s.log 2> %s/profile_%s_stderr.log'%(kernel_name,to_profile,profiling_results_name+kernel_name,kernel_name,profiling_results_name+kernel_name,kernel_name))
        print(color.BOLD+"Aggregating collected metrics"+color.END)
        os.system('bash ./tool_scripts/merge_ncu.sh %s %s > /dev/null' %(kernel_name,profiling_results_name))
        os.system('rm -f *.log')
        PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY = roofline.plot_gpu(kernel_name,"NCU",mode,version,json_file)
        print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Roofline has been generated, the results are in "+ color.PURPLE + cwd +"/"+profiling_results_name+kernel_name+color.END)
    return PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY

if __name__ == "__main__":
    paa.ascii_art()
    options = -1
    mode = 1 #SET TO 1 IN ORDER TO PLOT ALL PRECISIONS, 0 TO ONLY PLOT THE MOST RELEVANT ONE, 2 WHEN OPTIMIZING
    device_id = 0
    while (options!=0 and options!=1 and options!=2 and options!=3 and options!=4):
        print(color.BOLD+"Which operation would you like to perform?")
        print(color.CYAN + color.BOLD + "0) Tool pre-compute utilities (Tool compatibility checker, Makefile parser)" + color.END)
        print(color.YELLOW + color.BOLD +   "1) Print Roofline of a GPU kernel" + color.END)
        print(color.ORANGE + color.BOLD +   "2) Print Roofline of a GPU kernel - lazy option with txt (perform_roofline/roof.txt)" + color.END)
        print(color.BLUE + color.BOLD +     "3) Optimize your kernel using the tool" + color.END)
        print(color.PURPLE + color.BOLD +   "4) Optimize your kernel using the tool - lazy option with txt (perform_roofline/roof.txt)" + color.END)
        # print("2) Optimize your kernel using the tool")
        options = int(input())
        if(options!=0 and options!=1 and options!=2 and options!=3 and options!=4):
            print("Select one of the supported options")
    if (options == 0):
        inner_option = 0
        while (inner_option!=1 and inner_option!=2):
            print(color.BOLD+"Which operation would you like to perform?")
            print(color.CYAN + color.BOLD + "1) Check tool compatibility" + color.END)
            print(color.YELLOW + color.BOLD + "2) Parse Makefile for optimization operations" +color.RED+ " (Warning, this will add -lineinfo to all your Makefile compilers, it might break something)" + color.END)
            inner_option = int(input())
            if(inner_option == 1):
                admin_only_profile = float(os.popen('cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly | cut -d : -f 2-').read())
                mp_lib = os.popen('pip3 list | grep matplotlib').read()
                np_lib = os.popen('pip3 list | grep numpy').read()
                if admin_only_profile:
                    print(color.BOLD+color.RED+"Application can only be profiled by sudo users"+color.END)
                else:
                    print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Application can be profiled by non sudo users"+color.END)
                if not mp_lib:
                    print(color.BOLD+color.RED+"Cannot find matplotlib library in Python3, please install it"+color.END)
                    # module doesn't exist, deal with it.
                else:
                    print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Found matplotlib"+color.END)
                if not np_lib:
                    print(color.BOLD+color.RED+"Cannot find numpy library in Python3, please install it"+color.END)
                    # module doesn't exist, deal with it.
                else:
                    print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Found numpy"+color.END)
            if(inner_option==2):
                print(color.BOLD+"Insert the absolute path or relative path (from "+os.getcwd()+") to the Makefile you want to parse" +color.END)
                makefile_path = input()
                print(color.BOLD+"Parsing Makefile" +color.END)
                parse_makefile(makefile_path)                
            if(inner_option!=1 and inner_option!=2):
                print("Select one of the supported options")

    elif options==1:
        #this import is done here so that checks can be performed easily
        import tool_scripts.roofline_plot as roofline
        
        #ask for json file containing empirical gpu values
        json_answer = None
        while (json_answer!="yes" and json_answer!="no"):
            print(color.BOLD+"Do you have a JSON file containing empirical values to build the Roofline (see examples in the gpu_json folder)? (yes/no)"+color.END)
            json_answer = input()
        if (json_answer == "yes"):
            print(color.BOLD+"Insert the absolute path or relative path (from "+os.getcwd()+") to the JSON file" +color.END)
            json_file = input()
        else:
            print(color.BOLD+"Starlight will use theoretical values to build the Roofline"+color.END)
            json_file = None

        #ask for kernel and application info
        print(color.BOLD+"Insert the name of the kernel you want to plot"+color.END)
        kernel_name = input()
        print(color.BOLD+"Insert the absolute path or relative path (from "+os.getcwd()+") to the application you want to profile" +color.END)
        app_path = input()
        to_profile = app_path
        print(color.BOLD+"Insert application flags and/or arguments" +color.END)
        to_profile+=" "
        to_profile+= input()
        run_roofline(kernel_name, to_profile, app_path, mode, device_id, json_file)
        
    elif options==2:
        #this import is done here so that checks can be performed easily
        import tool_scripts.roofline_plot as roofline
        print(color.BOLD+"Reading from roof.txt in perform_roofline/"+color.END)
        json_answer = os.popen('sed -n "2p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        if (json_answer == "yes"):
            json_file = os.popen('sed -n "3p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        else:
            json_file = None
        kernel_name = os.popen('sed -n "4p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        app_path = os.popen('sed -n "5p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        to_profile = app_path
        to_profile += " " + os.popen('sed -n "6p" perform_roofline/roof.txt | cut -d : -f 2- | tr -d "\n"').read()
        run_roofline(kernel_name, to_profile, app_path, mode, device_id, json_file)

    elif options==3:
        #this import is done here so that checks can be performed easily
        import tool_scripts.roofline_plot as roofline

        #ask for json file containing empirical gpu values
        json_answer = None
        while (json_answer!="yes" and json_answer!="no"):
            print(color.BOLD+"Do you have a JSON file containing empirical values to build the Roofline (see examples in the gpu_json folder)? (yes/no)"+color.END)
            json_answer = input()
        if (json_answer == "yes"):
            print(color.BOLD+"Insert the absolute path or relative path (from "+os.getcwd()+") to the JSON file" +color.END)
            json_file = input()
        else:
            print(color.BOLD+"Starlight will use theoretical values to build the Roofline"+color.END)
            json_file = None

        #ask for kernel and application info
        print(color.BOLD+"Insert the name of the kernel you want to plot"+color.END)
        kernel_name = input()
        print(color.BOLD+"Insert the absolute path or relative path (from "+os.getcwd()+") to the application you want to profile" +color.END)
        app_path = input()
        to_profile = app_path
        print(color.BOLD+"Insert application flags and/or arguments" +color.END)
        to_profile+=" "
        to_profile+= input()
        PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY = run_roofline(kernel_name, to_profile, app_path, 2, device_id, json_file)
        build_sampler()
        kernel_sample(kernel_name, app_path, to_profile)
        ops.generate_optimization_report(kernel_name, app_path, to_profile, PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY)
    elif options==4:
        #this import is done here so that checks can be performed easily
        import tool_scripts.roofline_plot as roofline
        print(color.BOLD+"Reading from roof.txt in perform_roofline/"+color.END)
        json_answer = os.popen('sed -n "2p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        if (json_answer == "yes"):
            json_file = os.popen('sed -n "3p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        else:
            json_file = None
        kernel_name = os.popen('sed -n "4p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        app_path = os.popen('sed -n "5p" perform_roofline/roof.txt | cut -d : -f 2-| tr -d "\n"').read()
        to_profile = app_path
        to_profile += " " + os.popen('sed -n "6p" perform_roofline/roof.txt | cut -d : -f 2- | tr -d "\n"').read()
        PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY = run_roofline(kernel_name, to_profile, app_path, 2, device_id, json_file)
        build_sampler()
        kernel_sample(kernel_name, app_path, to_profile)
        ops.generate_optimization_report(kernel_name, app_path, to_profile, PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY)
        # print(PERFORMANCE, OPERATIONAL_INTENSITY, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, SHARED_UT, SM_EFFICIENCY)