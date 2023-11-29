#!/bin/bash

MKL_TOPLEVEL=/opt/intel/oneapi/mkl/latest
LLVM_BUILD_LIB=$HOME/llvm-project/build/lib

MKL_LIB=$MKL_TOPLEVEL/lib
MKL_INC=$MKL_TOPLEVEL/include


if ! [ -d $MKL_TOPLEVEL ] ; then
    echo [ERROR] MKL_TOPLEVEL "'"$MKL_TOPLEVEL"'" does not exist
    exit 1
fi

if ! [ -d $MKL_LIB ] ; then
    echo [ERROR] MKL_LIB "'"$MKL_LIB"'" does not exist
    exit 1
fi

if ! [ -d $MKL_INC ] ; then
    echo [ERROR] MKL_INC "'"$MKL_INC"'" does not exist
    exit 1
fi

if ! [ -d $LLVM_BUILD_LIB ] ; then
    echo [ERROR] LLVM_BUILD_LIB "'"$LLVM_BUILD_LIB"'" does not exist
    exit 1
fi

export LD_LIBRARY_PATH="$MKL_TOPLEVEL/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HOME/llvm-project/build/lib/":$LD_LIBRARY_PATH


g++ mkl_gemm.cpp -g -L$MKL_LIB -L $LLVM_BUILD_LIB -I$MKL_INC -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmlir_c_runner_utils && \
    taskset --cpu-list 0 ./a.out
