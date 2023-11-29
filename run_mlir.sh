#!/bin/bash

LLVM_BUILD=$HOME/llvm-project/build/

MLIR_BIN=$LLVM_BUILD/bin
MLIR_LIB=$LLVM_BUILD/lib

if ! [ -d $LLVM_BUILD ] ; then
    echo [ERROR] LLVM_BUILD "'"$LLVM_BUILD"'" does not exist
    exit 1
fi

if ! [ -d $MLIR_BIN ] ; then
    echo [ERROR] MLIR_BIN "'"$MLIR_BIN"'" does not exist
    exit 1
fi

if ! [ -d $MLIR_LIB ] ; then
    echo [ERROR] MLIR_LIB "'"$MLIR_LIB"'" does not exist
    exit 1
fi

export PATH="$MLIR_BIN":$PATH

echo Base
mlir-opt linalg_gemm.mlir -pass-pipeline="builtin.module(func-bufferize,func.func(linalg-bufferize,convert-linalg-to-affine-loops),convert-vector-to-scf,convert-linalg-to-loops,lower-affine,convert-scf-to-cf,canonicalize,cse,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)" | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=$MLIR_LIB/libmlir_c_runner_utils.so
echo -e "\n"

echo Unroll and jam
mlir-opt linalg_gemm.mlir -pass-pipeline="builtin.module(func-bufferize,func.func(linalg-bufferize,convert-linalg-to-affine-loops,affine-loop-unroll-jam{unroll-jam-factor=8}),convert-vector-to-scf,convert-linalg-to-loops,lower-affine,convert-scf-to-cf,canonicalize,cse,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)" | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=$MLIR_LIB/libmlir_c_runner_utils.so
echo -e "\n"

echo Vectorizing
mlir-opt linalg_gemm.mlir -pass-pipeline="builtin.module(func-bufferize,func.func(linalg-bufferize,convert-linalg-to-affine-loops,affine-super-vectorize{virtual-vector-size=1024}),convert-vector-to-scf,convert-linalg-to-loops,lower-affine,convert-scf-to-cf,canonicalize,cse,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)" | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=$MLIR_LIB/libmlir_c_runner_utils.so
echo -e "\n"

echo Cache tiling
mlir-opt linalg_gemm.mlir -pass-pipeline="builtin.module(func-bufferize,func.func(linalg-bufferize,convert-linalg-to-affine-loops,affine-loop-tile{tile-size=16}),convert-vector-to-scf,convert-linalg-to-loops,lower-affine,convert-scf-to-cf,canonicalize,cse,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)" | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=$MLIR_LIB/libmlir_c_runner_utils.so
echo -e "\n"

echo Cache tiling best
mlir-opt linalg_gemm.mlir -pass-pipeline="builtin.module(func-bufferize,func.func(linalg-bufferize,convert-linalg-to-affine-loops,affine-loop-tile{tile-sizes=64,32,8}),convert-vector-to-scf,convert-linalg-to-loops,lower-affine,convert-scf-to-cf,canonicalize,cse,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts)" | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=$MLIR_LIB/libmlir_c_runner_utils.so
echo -e "\n"
