```
   ______________    ____  __    ____________  ________ 
  / ___/_  __/   |  / __ \/ /   /  _/ ____/ / / /_  __/ 
  \__ \ / / / /| | / /_/ / /    / // / __/ /_/ / / /   
 ___/ // / / ___ |/ _, _/ /____/ // /_/ / __  / / /     
/____//_/ /_/  |_/_/ |_/_____/___/\____/_/ /_/ /_/      
```
## Introduction
<p align="justify">
Over the past few years, GPUs have found widespread adoption in many scientific domains, offering notable performance and energy efficiency advantages compared to CPUs. However, optimizing GPU high-performance kernels poses challenges given the complexities of GPU architectures and programming models. Moreover, current GPU development tools provide few high-level suggestions and overlook the underlying hardware. Here we present Starlight, an open-source, highly flexible tool for enhancing GPU kernel analysis and optimization. Starlight autonomously describes Roofline Models, examines performance metrics, and correlates these insights with GPU architectural bottlenecks. Additionally, Starlight predicts potential performance enhancements before altering the source code. We demonstrate its efficacy by applying it to literature genomics and physics applications, attaining speedups from 1.1× to 2.5× over state-of-the-art baselines. Furthermore, Starlight supports the development of new GPU kernels, which we exemplify through an image processing application, showing speedups of 12.7× and 140× when compared against state-of-the-art FPGA- and GPU-based solutions.
</p>

## Requirements (for sys admin)
<p align="justify">
The tool requires Python3 and pip3 and the NIVDIA CUDA toolkit in order to be used.

Python3 and pip3 can be installed using your packet manager e.g., for Ubuntu:
```
sudo apt-get install python3
sudo apt-get -y install python3-pip
```
or for CentOS/RHEL
```
sudo yum install python3
sudo yum -y install python3-pip
```

We advise to install CUDA by following the guide available here:
https://developer.nvidia.com/cuda-downloads
**Please follow the NVIDIA guide to install CUDA.**

By installing CUDA following this guide every package required by the tool will be installed.
Moreover the users must have the permissions to profile the GPU applications on the machine.

To check if you have the necessary permissions either run the tool and choose option ```0``` and then check tool compatibility, in this case the tool itself will check if you have the necessary permissions, this will work only if Python3 is already installed in the system.

Alternatively you can simply type:
```
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```
If the output of this command is ```0``` all users can profile GPU applications on the system.
Otherwise profiling permissions can be easily added by the system administrator by typing:
```
(echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"') | sudo tee -a /etc/modprobe.d/RestrictedProfiling.conf >/dev/null
```
</p>

## Requirements (for regular users)
<p align="justify">

The tool requires two python3 packages that can be installed without root permissions.
First, check if you already have these packages by running the tool using option ```0``` and then check tool compatibility.
If some of the requirements regarding packages are not met, simply install the packages using the provided **requirements.txt** file by typing:

```
pip3 install -r requirements.txt
```
</p>

------------------------------------------------------------------------


## Usage
<p align="justify">

To run the tool simply type:

```
python3 starlight.py
```
Within the tool you can choose multiple options
### <a name="precheck"></a>Option 0 (Pre-exec-checks) 

The tool provides two suboptions in this scenario, suboption **1** will check the tool compatibility, checking if the necessary python packages and profiling options are enabled.
Suboption **2** will offer you the possibility to parse your application makefile to add the ```-lineinfo``` flag to the compiler options. This flag is necessary to retrieve the necessary information by the code sampler and thus to perform code optimizations suggestions. You can add this flag manually to your makefile if you desire or just simply add it to the compiler flags if you compile your executables manually. Remember, this flag is **NOT** necessary if you only want to create the roofline for application.

### <a name="roofpltint"></a>Option 1 (Roofline-Plot Interactive)
This option will plot the roofline model for your desired kernel and application.
In this option the user will be asked to input the kernel name, the application path and the application command line options.
After the kernel has been profiled, a roofline model will be generated into the following folder:
```
roofline_results/KERNEL_NAME/roofline-KERNEL_NAME.pdf
```
### <a name="roofplt"></a>Option 2 (Roofline-Plot)
This option mirrors **Option 1** and plots the roofline model for your desired kernel and application, but does not require the user interaction.
Instead, by choosing this option the tool will read the information regarding kernel name, the application path and the application command line options from the following file:
```
perform_roofline/roof.txt
```
Therefore, if one whishes to modify some options they will need to modify that file.
After the kernel has been profiled, a roofline model will be generated into the following folder:
```
roofline_results/KERNEL_NAME/roofline-KERNEL_NAME.pdf
```

### <a name="optint"></a>Option 3 (Roofline-Plot Interactive)
This option will plot the roofline model, analyze the code and provide optimizations suggestions for your desired kernel and application.
In this option the user will be asked to input the kernel name, the application path and the application command line options.
After the kernel has been profiled and sampled, a roofline model will be generated into the following folder:
```
optimization_results/KERNEL_NAME/roofline-KERNEL_NAME.pdf
```
Moreover a list of suggestions for the kernel optimizations will be provided in the same folder in txt format:
```
optimization_results/KERNEL_NAME/optimizations-KERNEL_NAME.txt
```
### <a name="opt"></a>Option 4 (Roofline-Plot)
Like **Option 2** this option mirrors **Option 3** and plots the roofline model, analyzes the code and provide optimizations suggestions for your desired kernel and application, but does not require the user interaction.
Instead, by choosing this option the tool will read the information regarding kernel name, the application path and the application command line options from the following file:
```
perform_optimization/opt.txt
```
Therefore, if one whishes to modify some options they will need to modify that file.
After the kernel has been profiled and sampled, a roofline model will be generated into the following folder:
```
optimization_results/KERNEL_NAME/roofline-KERNEL_NAME.pdf
```
Moreover a list of suggestions for the kernel optimizations will be provided in the same folder in txt format:
```
optimization_results/KERNEL_NAME/optimizations-KERNEL_NAME.txt
```
</p>

------------------------------------------------------------------------

### NBody Demo
The tool contains an n-body demo where you may find an out-of-the-box example.
The code comes from NVIDIA [CUDA-samples](https://github.com/NVIDIA/cuda-samples).


#### Demo - Step 0: Setup 
Execute the tool with ```python starlight.py``` or ```python3 starlight.py```

Verify your setup following [Option 0 - suboption 1](#precheck).
Then apply [the second suboption](#precheck) to  ```examples/nbody/Makefile```
You will find a new `Makefile_modded` to perform fine-grained profiling.

#### Demo - Step 1: Roofline Plot
Execute the tool again ```python starlight.py```

Pick [Option 1](#roofpltint) and input
```integrateBodies``` as Kernel name
```./examples/nbody/nbody``` as executable name and path
```-benchmark``` as flag/input arguments

As alternative execution the tool offer the chance to execute in batch mode providing a configuration file in the `perform_roofline/` folder.
To do this execute the tool and pick [Option 2](#roofplt)

The file looks as follows:
```
### Execute file with the information below
kernel:integrateBodies
app:./examples/nbody/nbody
flags:-benchmark
```

The tool will create a new folder called `roofline_results/integrateBodies` where you will find the Roofline of the application and other logs.

#### Demo - Step 2: Optimization Report
Execute the tool again  ```python starlight.py```

Pick [Option 3](#optint) and input
```integrateBodies``` as Kernel name
```./examples/nbody/nbody``` as executable name and path
```-benchmark``` as flag/input arguments

As alternative execution the tool offer the chance to execute in batch mode providing a configuration file in the `perform_optimization/` folder.
To do this execute the tool and pick [Option 4](#opt)
The file looks as follows:
```
### Execute file with the information below
kernel:integrateBodies
app:./examples/nbody/nbody
flags:-benchmark
```

The tool will create a new folder called `optimization_results/integrateBodies` where you will find the log of the possible optimizations, the Roofline of the application, and other logs.

------------------------------------------------------------------------

## Citation

To cite our work or to know more about our methods, please refer to:
```
@article{zeni2023starlight,
  title={Starlight: A kernel optimizer for GPU processing},
  author={Zeni, Alberto and Del Sozzo, Emanuele and D'Arnese, Eleonora and Conficconi, Davide and Santambrogio, Marco D},
  journal={Journal of Parallel and Distributed Computing},
  pages={104832},
  year={2023},
  publisher={Elsevier}
}
```

## Acknowledgments

Data used in this publication were generated by the National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The Authors would like to thank the NVIDIA University Program for the hardware donations and Oracle Research Program for the Oracle Cloud Credits.
