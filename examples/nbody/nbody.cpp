/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bodysystemcuda.h"
#include "bodysystemcpu.h"
#include "cuda_runtime.h"

bool benchmark = false;
bool compareToCPU = false;
bool QATest = false;
int blockSize = 256;
bool useHostMem = false;
bool useP2P = true;  // this is always optimal to use P2P path when available
bool fp64 = false;
bool useCpu = false;
int numDevsRequested = 1;
bool bPause = false;
bool bDispInteractions = false;
bool bSupportDouble = false;
int flopsPerInteraction = 20;

char deviceName[100];

enum { M_VIEW = 0, M_MOVE };

int numBodies = 16384;

std::string tipsyFile = "";

int numIterations = 0;  // run until exit

void computePerfStats(double &interactionsPerSecond, double &gflops,
                      float milliseconds, int iterations) {
  // double precision uses intrinsic operation followed by refinement,
  // resulting in higher operation count per interaction.
  // (Note Astrophysicists use 38 flops per interaction no matter what,
  // based on "historical precedent", but they are using FLOP/s as a
  // measure of "science throughput". We are using it as a measure of
  // hardware throughput.  They should really use interactions/s...
  // const int flopsPerInteraction = fp64 ? 30 : 20;
  interactionsPerSecond = (float)numBodies * (float)numBodies;
  interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
  gflops = interactionsPerSecond * (float)flopsPerInteraction;
}

////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams {
  float m_timestep;
  float m_clusterScale;
  float m_velocityScale;
  float m_softening;
  float m_damping;
  float m_pointSize;
  float m_x, m_y, m_z;

  void print() {
    printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", m_timestep,
           m_clusterScale, m_velocityScale, m_softening, m_damping, m_pointSize,
           m_x, m_y, m_z);
  }
};

NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0,
     -50},
};

int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
bool cycleDemo = true;
int activeDemo = 0;
float demoTime = 10000.0f;  // ms
StopWatchInterface *demoTimer = NULL, *timer = NULL;

// run multiple iterations to compute an average sort time

NBodyParams activeParams = demoParams[activeDemo];

// fps
cudaEvent_t startEvent, stopEvent;
cudaEvent_t hostMemSyncEvent;

template <typename T>
class NBodyDemo {
 public:
  static void Create() { m_singleton = new NBodyDemo; }
  static void Destroy() { delete m_singleton; }

  static void init(int numBodies, int numDevices, int blockSize, bool usePBO,
                   bool useHostMem, bool useP2P, bool useCpu, int devID) {
    m_singleton->_init(numBodies, numDevices, blockSize, usePBO, useHostMem,
                       useP2P, useCpu, devID);
  }

  static void reset(int numBodies, NBodyConfig config) {
    m_singleton->_reset(numBodies, config);
  }

  //static void selectDemo(int index) { m_singleton->_selectDemo(index); }

  static bool compareResults(int numBodies) {
    return m_singleton->_compareResults(numBodies);
  }

  static void runBenchmark(int iterations) {
    m_singleton->_runBenchmark(iterations);
  }

  static void updateParams() {
    m_singleton->m_nbody->setSoftening(activeParams.m_softening);
    m_singleton->m_nbody->setDamping(activeParams.m_damping);
  }

  static void updateSimulation() {
    m_singleton->m_nbody->update(activeParams.m_timestep);
  }

  static void getArrays(T *pos, T *vel) {
    T *_pos = m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION);
    T *_vel = m_singleton->m_nbody->getArray(BODYSYSTEM_VELOCITY);
    memcpy(pos, _pos, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
    memcpy(vel, _vel, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
  }

  static void setArrays(const T *pos, const T *vel) {
    if (pos != m_singleton->m_hPos) {
      memcpy(m_singleton->m_hPos, pos, numBodies * 4 * sizeof(T));
    }

    if (vel != m_singleton->m_hVel) {
      memcpy(m_singleton->m_hVel, vel, numBodies * 4 * sizeof(T));
    }

    m_singleton->m_nbody->setArray(BODYSYSTEM_POSITION, m_singleton->m_hPos);
    m_singleton->m_nbody->setArray(BODYSYSTEM_VELOCITY, m_singleton->m_hVel);

  }

 private:
  static NBodyDemo *m_singleton;

  BodySystem<T> *m_nbody;
  BodySystemCUDA<T> *m_nbodyCuda;
  BodySystemCPU<T> *m_nbodyCpu;

  T *m_hPos;
  T *m_hVel;
  float *m_hColor;

 private:
  NBodyDemo()
      : m_nbody(0),
        m_nbodyCuda(0),
        m_nbodyCpu(0),
        m_hPos(0),
        m_hVel(0),
        m_hColor(0) {}

  ~NBodyDemo() {
    if (m_nbodyCpu) {
      delete m_nbodyCpu;
    }

    if (m_nbodyCuda) {
      delete m_nbodyCuda;
    }

    if (m_hPos) {
      delete[] m_hPos;
    }

    if (m_hVel) {
      delete[] m_hVel;
    }

    if (m_hColor) {
      delete[] m_hColor;
    }

    sdkDeleteTimer(&demoTimer);

  }

  void _init(int numBodies, int numDevices, int blockSize, bool bUsePBO,
             bool useHostMem, bool useP2P, bool useCpu, int devID) {
    if (useCpu) {
      m_nbodyCpu = new BodySystemCPU<T>(numBodies);
      m_nbody = m_nbodyCpu;
      m_nbodyCuda = 0;
    } else {
      m_nbodyCuda = new BodySystemCUDA<T>(numBodies, numDevices, blockSize,
                                          bUsePBO, useHostMem, useP2P, devID);
      m_nbody = m_nbodyCuda;
      m_nbodyCpu = 0;
    }

    // allocate host memory
    m_hPos = new T[numBodies * 4];
    m_hVel = new T[numBodies * 4];
    m_hColor = new float[numBodies * 4];

    m_nbody->setSoftening(activeParams.m_softening);
    m_nbody->setDamping(activeParams.m_damping);

    if (useCpu) {
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventCreate(&startEvent));
      checkCudaErrors(cudaEventCreate(&stopEvent));
      checkCudaErrors(cudaEventCreate(&hostMemSyncEvent));
    }

    sdkCreateTimer(&demoTimer);
    sdkStartTimer(&demoTimer);
  }

  void _reset(int numBodies, NBodyConfig config) {
    if (tipsyFile == "") {
      randomizeBodies(config, m_hPos, m_hVel, m_hColor,
                      activeParams.m_clusterScale, activeParams.m_velocityScale,
                      numBodies, true);
      setArrays(m_hPos, m_hVel);
    } else {
      m_nbody->loadTipsyFile(tipsyFile);
      ::numBodies = m_nbody->getNumBodies();
    }
  }

  bool _compareResults(int numBodies) {
    assert(m_nbodyCuda);

    bool passed = true;

    m_nbody->update(0.001f);

    {
      m_nbodyCpu = new BodySystemCPU<T>(numBodies);

      m_nbodyCpu->setArray(BODYSYSTEM_POSITION, m_hPos);
      m_nbodyCpu->setArray(BODYSYSTEM_VELOCITY, m_hVel);

      m_nbodyCpu->update(0.001f);

      T *cudaPos = m_nbodyCuda->getArray(BODYSYSTEM_POSITION);
      T *cpuPos = m_nbodyCpu->getArray(BODYSYSTEM_POSITION);

      T tolerance = 0.0005f;

      for (int i = 0; i < numBodies; i++) {
        if (fabs(cpuPos[i] - cudaPos[i]) > tolerance) {
          passed = false;
          printf("Error: (host)%f != (device)%f\n", cpuPos[i], cudaPos[i]);
        }
      }
    }
    if (passed) {
      printf("  OK\n");
    }
    return passed;
  }

  void _runBenchmark(int iterations) {
    // once without timing to prime the device
    if (!useCpu) {
      m_nbody->update(activeParams.m_timestep);
    }

    if (useCpu) {
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventRecord(startEvent, 0));
    }

    for (int i = 0; i < iterations; ++i) {
      m_nbody->update(activeParams.m_timestep);
    }

    float milliseconds = 0;

    if (useCpu) {
      sdkStopTimer(&timer);
      milliseconds = sdkGetTimerValue(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventRecord(stopEvent, 0));
      checkCudaErrors(cudaEventSynchronize(stopEvent));
      checkCudaErrors(
          cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    }

    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds, iterations);

    printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies,
           iterations, milliseconds);
    printf("= %.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n", gflops,
           (sizeof(T) > 4) ? "double" : "single", flopsPerInteraction);
  }
};

void finalize() {
  if (!useCpu) {
    checkCudaErrors(cudaEventDestroy(startEvent));
    checkCudaErrors(cudaEventDestroy(stopEvent));
    checkCudaErrors(cudaEventDestroy(hostMemSyncEvent));
  }

  NBodyDemo<float>::Destroy();

  if (bSupportDouble) NBodyDemo<double>::Destroy();
}

template <>
NBodyDemo<double> *NBodyDemo<double>::m_singleton = 0;
template <>
NBodyDemo<float> *NBodyDemo<float>::m_singleton = 0;

template <typename T_new, typename T_old>
void switchDemoPrecision() {
  cudaDeviceSynchronize();

  fp64 = !fp64;
  flopsPerInteraction = fp64 ? 30 : 20;

  T_old *oldPos = new T_old[numBodies * 4];
  T_old *oldVel = new T_old[numBodies * 4];

  NBodyDemo<T_old>::getArrays(oldPos, oldVel);

  // convert float to double
  T_new *newPos = new T_new[numBodies * 4];
  T_new *newVel = new T_new[numBodies * 4];

  for (int i = 0; i < numBodies * 4; i++) {
    newPos[i] = (T_new)oldPos[i];
    newVel[i] = (T_new)oldVel[i];
  }

  NBodyDemo<T_new>::setArrays(newPos, newVel);

  cudaDeviceSynchronize();

  delete[] oldPos;
  delete[] oldVel;
  delete[] newPos;
  delete[] newVel;
}

void updateSimulation() {
  if (fp64) {
    NBodyDemo<double>::updateSimulation();
  } else {
    NBodyDemo<float>::updateSimulation();
  }
}

void updateParams() {
  if (fp64) {
    NBodyDemo<double>::updateParams();
  } else {
    NBodyDemo<float>::updateParams();
  }
}

void showHelp() {
  printf(
      "\t-fp64             (use double precision floating point values for "
      "simulation)\n");
  printf("\t-hostmem          (stores simulation data in host memory)\n");
  printf("\t-benchmark        (run benchmark to measure performance) \n");
  printf(
      "\t-numbodies=<N>    (number of bodies (>= 1) to run in simulation) \n");
  printf(
      "\t-device=<d>       (where d=0,1,2.... for the CUDA device to use)\n");
  printf(
      "\t-numdevices=<i>   (where i=(number of CUDA devices > 0) to use for "
      "simulation)\n");
  printf(
      "\t-compare          (compares simulation results running once on the "
      "default GPU and once on the CPU)\n");
  printf("\t-cpu              (run n-body simulation on the CPU)\n");
  printf("\t-tipsy=<file.bin> (load a tipsy model file for simulation)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  bool bTestResults = true;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("\n> Command line options\n");
    showHelp();
    return 0;
  }

  printf(
      "Run \"nbody -benchmark [-numbodies=<numBodies>]\" to measure "
      "performance.\n");
  showHelp();

  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  benchmark = (checkCmdLineFlag(argc, (const char **)argv, "benchmark") != 0);

  compareToCPU =
      ((checkCmdLineFlag(argc, (const char **)argv, "compare") != 0) ||
       (checkCmdLineFlag(argc, (const char **)argv, "qatest") != 0));

  QATest = (checkCmdLineFlag(argc, (const char **)argv, "qatest") != 0);
  useHostMem = (checkCmdLineFlag(argc, (const char **)argv, "hostmem") != 0);
  fp64 = (checkCmdLineFlag(argc, (const char **)argv, "fp64") != 0);

  flopsPerInteraction = fp64 ? 30 : 20;

  useCpu = (checkCmdLineFlag(argc, (const char **)argv, "cpu") != 0);

  if (checkCmdLineFlag(argc, (const char **)argv, "numdevices")) {
    numDevsRequested =
        getCmdLineArgumentInt(argc, (const char **)argv, "numdevices");

    if (numDevsRequested < 1) {
      printf(
          "Error: \"number of CUDA devices\" specified %d is invalid.  Value "
          "should be >= 1\n",
          numDevsRequested);
      exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
    } else {
      printf("number of CUDA devices  = %d\n", numDevsRequested);
    }
  }

  int numDevsAvailable = 0;
  bool customGPU = false;
  cudaGetDeviceCount(&numDevsAvailable);

  if (numDevsAvailable < numDevsRequested) {
    printf("Error: only %d Devices available, %d requested.  Exiting.\n",
           numDevsAvailable, numDevsRequested);
    exit(EXIT_FAILURE);
  }

  if (numDevsRequested > 1) {
    // If user did not explicitly request host memory to be used, we default to
    // P2P.
    // We fallback to host memory, if any of GPUs does not support P2P.
    bool allGPUsSupportP2P = true;
    if (!useHostMem) {
      // Enable P2P only in one direction, as every peer will access gpu0
      for (int i = 1; i < numDevsRequested; ++i) {
        int canAccessPeer;
        checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, i, 0));

        if (canAccessPeer != 1) {
          allGPUsSupportP2P = false;
        }
      }

      if (!allGPUsSupportP2P) {
        useHostMem = true;
        useP2P = false;
      }
    }
  }

  printf("> Simulation data stored in %s memory\n",
         useHostMem ? "system" : "video");
  printf("> %s precision floating point simulation\n",
         fp64 ? "Double" : "Single");
  printf("> %d Devices used for simulation\n", numDevsRequested);

  int devID;
  cudaDeviceProp props;

  if (useCpu) {
    useHostMem = true;
    compareToCPU = false;
    bSupportDouble = true;

#ifdef OPENMP
    printf("> Simulation with CPU using OpenMP\n");
#else
    printf("> Simulation with CPU\n");
#endif
  }

  
  if (!useCpu) {
    // Now choose the CUDA Device
    if (benchmark || compareToCPU || useHostMem) {

      if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        customGPU = true;
      }

      devID = findCudaDevice(argc, (const char **)argv);
    } else 
    {
      if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        customGPU = true;
      }

      devID = findCudaDevice(argc, (const char **)argv);
    }

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    bSupportDouble = true;

#if CUDART_VERSION < 4000

    if (numDevsRequested > 1) {
      printf("MultiGPU n-body requires CUDA 4.0 or later\n");
      exit(EXIT_SUCCESS);
    }

#endif

    // Initialize devices
    if (numDevsRequested > 1 && customGPU) {
      printf("You can't use --numdevices and --device at the same time.\n");
      exit(EXIT_SUCCESS);
    }

    if (customGPU || numDevsRequested == 1) {
      cudaDeviceProp props;
      checkCudaErrors(cudaGetDeviceProperties(&props, devID));
      printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor,
             props.name);
    } else {
      for (int i = 0; i < numDevsRequested; i++) {
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));

        printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor,
               props.name);

        if (useHostMem) {
#if CUDART_VERSION >= 2020

          if (!props.canMapHostMemory) {
            fprintf(stderr, "Device %d cannot map host memory!\n", devID);
            exit(EXIT_SUCCESS);
          }

          if (numDevsRequested > 1) {
            checkCudaErrors(cudaSetDevice(i));
          }

          checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
          fprintf(stderr,
                  "This CUDART version does not support "
                  "<cudaDeviceProp.canMapHostMemory> field\n");
          exit(EXIT_SUCCESS);
#endif
        }
      }

      // CC 1.2 and earlier do not support double precision
      if (props.major * 10 + props.minor <= 12) {
        bSupportDouble = false;
      }
    }

    // if(numDevsRequested > 1)
    //    checkCudaErrors(cudaSetDevice(devID));

    if (fp64 && !bSupportDouble) {
      fprintf(stderr,
              "One or more of the requested devices does not support double "
              "precision floating-point\n");
      exit(EXIT_SUCCESS);
    }
  }

  numIterations = 0;
  blockSize = 0;

  if (checkCmdLineFlag(argc, (const char **)argv, "i")) {
    numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "i");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "blockSize")) {
    blockSize = getCmdLineArgumentInt(argc, (const char **)argv, "blockSize");
  }

  if (blockSize == 0)  // blockSize not set on command line
    blockSize = 256;

  // default number of bodies is #SMs * 4 * CTA size
  if (useCpu)
#ifdef OPENMP
    numBodies = 8192;

#else
    numBodies = 4096;
#endif
  else if (numDevsRequested == 1) {
    numBodies = compareToCPU ? 4096 : blockSize * 4 * props.multiProcessorCount;
  } else {
    numBodies = 0;

    for (int i = 0; i < numDevsRequested; i++) {
      cudaDeviceProp props;
      checkCudaErrors(cudaGetDeviceProperties(&props, i));
      numBodies +=
          blockSize * (props.major >= 2 ? 4 : 1) * props.multiProcessorCount;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "numbodies")) {
    numBodies = getCmdLineArgumentInt(argc, (const char **)argv, "numbodies");

    if (numBodies < 1) {
      printf(
          "Error: \"number of bodies\" specified %d is invalid.  Value should "
          "be >= 1\n",
          numBodies);
      exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
    } else if (numBodies % blockSize) {
      int newNumBodies = ((numBodies / blockSize) + 1) * blockSize;
      printf(
          "Warning: \"number of bodies\" specified %d is not a multiple of "
          "%d.\n",
          numBodies, blockSize);
      printf("Rounding up to the nearest multiple: %d.\n", newNumBodies);
      numBodies = newNumBodies;
    } else {
      printf("number of bodies = %d\n", numBodies);
    }
  }

  if (numBodies <= 1024) {
    activeParams.m_clusterScale = 1.52f;
    activeParams.m_velocityScale = 2.f;
  } else if (numBodies <= 2048) {
    activeParams.m_clusterScale = 1.56f;
    activeParams.m_velocityScale = 2.64f;
  } else if (numBodies <= 4096) {
    activeParams.m_clusterScale = 1.68f;
    activeParams.m_velocityScale = 2.98f;
  } else if (numBodies <= 8192) {
    activeParams.m_clusterScale = 1.98f;
    activeParams.m_velocityScale = 2.9f;
  } else if (numBodies <= 16384) {
    activeParams.m_clusterScale = 1.54f;
    activeParams.m_velocityScale = 8.f;
  } else if (numBodies <= 32768) {
    activeParams.m_clusterScale = 1.44f;
    activeParams.m_velocityScale = 11.f;
  }

  // Create the demo -- either double (fp64) or float (fp32, default)
  // implementation
  NBodyDemo<float>::Create();

  NBodyDemo<float>::init(numBodies, numDevsRequested, blockSize,
                         !(benchmark || compareToCPU || useHostMem), useHostMem,
                         useP2P, useCpu, devID);
  NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);

  if (bSupportDouble) {
    NBodyDemo<double>::Create();
    NBodyDemo<double>::init(numBodies, numDevsRequested, blockSize,
                            !(benchmark || compareToCPU || useHostMem),
                            useHostMem, useP2P, useCpu, devID);
    NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_SHELL);
  }

  if (fp64) {
    if (benchmark) {
      if (numIterations <= 0) {
        numIterations = 10;
      } else if (numIterations > 10) {
        printf("Advisory: setting a high number of iterations\n");
        printf("in benchmark mode may cause failure on Windows\n");
        printf("Vista and Win7. On these OSes, set iterations <= 10\n");
      }

      NBodyDemo<double>::runBenchmark(numIterations);
    } else if (compareToCPU) {
      bTestResults = NBodyDemo<double>::compareResults(numBodies);
    }

  } else {
    if (benchmark) {
      if (numIterations <= 0) {
        numIterations = 10;
      }

      NBodyDemo<float>::runBenchmark(numIterations);
    } else if (compareToCPU) {
      bTestResults = NBodyDemo<float>::compareResults(numBodies);
    } 
  }

  finalize();
  exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
