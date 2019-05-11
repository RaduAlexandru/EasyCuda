#pragma once

//pytorch
#include <c10/cuda/CUDACachingAllocator.h>

//cuda 
#include <cuda.h>

//eigen
#include <Eigen/Dense>

//c++
#include <iostream>

//my stuff
#include "surfel_renderer/utils/MiscUtils.h"

//opencv
#include "opencv2/opencv.hpp"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason


// #define CUD_C(stmt) do { 
//     stmt;
//     gpuAssert(#stmt, __FILE__, __LINE__);	\
// } while (0)
inline void gpuAssert(const char* stmt, const char* fname, int line)
{
   auto code = cudaGetLastError();
   if (code != cudaSuccess) 
   {
      printf("CUDA error %s, at %s:%i - for %s.\n", cudaGetErrorString(code), fname, line, stmt);
      exit(1);
   }
}
// CUD_C Check Macro. Will terminate the program if a error is detected.
#define CUDA_CHECK_ERROR() { gpuAssert("none", __FILE__, __LINE__);	}

