NVCC := nvcc
CUDA_INSTALL_PATH := $(shell which nvcc | sed 's/.........$$//')#double $ because of makefile
INCLUDES := -I$(CUDA_INSTALL_PATH)/include 

LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
LIBS := -L $(LIB_PATH) -lcuda -lcupti -lpcsamplingutil
LIBNAME := libpc_sampling_continuous.so
NVCCFLAGS := -Xcompiler -fPIC


all: pc_sampling_continuous pc_sampling_utility
pc_sampling_utility: pc_sampling_utility.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o pc_sampling_utility $< $(LIBS)
pc_sampling_continuous: pc_sampling_continuous.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(LIBNAME) -shared $< $(LIBS)
clean:
	rm -rf $(LIBNAME)
	rm -rf pc_sampling_utility
