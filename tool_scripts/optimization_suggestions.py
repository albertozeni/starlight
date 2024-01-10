import os
from random import sample
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

def generate_optimization_report(kernel_name, app_path, to_profile, PEAK, PERFORMANCE, OPERATIONAL_INTENSITY, GPU_GM_BANDWIDTH, RIDGE_POINT, DEVICE_UT, BRANCH_EFF, WARP_NON_PRED_EFF, SHARED_UT, SM_EFFICIENCY):
   cwd = os.getcwd()
   sample_path= cwd+"/optimization_results/"+kernel_name+"/sample_raw.csv"
   print(color.BOLD+"Generating optimization report"+color.END)
   print("PEAK: "+str(PEAK))
   print("GFLOPS/GIOPS: "+str(PERFORMANCE))
   print("OI: "+str(OPERATIONAL_INTENSITY))
   if(OPERATIONAL_INTENSITY<RIDGE_POINT):
      print("RELATIVE OI PEAK :"+str(GPU_GM_BANDWIDTH*OPERATIONAL_INTENSITY))
   else:
      print("RELATIVE OI PEAK :"+str(PEAK))
   print("OI RIDGE_POINT: "+str(RIDGE_POINT))
   print("DEVICE_UTILIZATION :"+str(DEVICE_UT))
   print("BRANCH_EFFICIENCY :"+str(BRANCH_EFF))
   print("NON_PRED_SM_EFFICIENCY :"+str(WARP_NON_PRED_EFF))
   print("SHARED_MEM_UTIL :"+str(SHARED_UT))
   print("SM_EFFICIENCY :"+str(SM_EFFICIENCY))

   gpu_sample = open(sample_path, 'r')
   print(color.BOLD+color.GREEN+u'\u2713'+color.END+color.BOLD+" Optimization analysis performed, the report has been generated in "+ color.BLUE+ cwd +"/optimization_results/"+kernel_name+color.END)                  
   gpu_sample.close()                                                   