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

inline void cuda_error_check(){

    auto code = cudaGetLastError();
    if(cudaSuccess != code){
        fprintf(stderr,"GPU Error: %s at %s:%i \n", cudaGetErrorString(code),  __FILE__, __LINE__ );
        exit(code);
    }
}
