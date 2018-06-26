#!/bin/bash

nvcc -ccbin=g++-5 -arch=sm_50 -o t1290 t1290.cu
# Check memory
cuda-memcheck ./t1290

