func.func @fc_relu(%lhs: memref<1024x1024xf32>, %rhs: memref<1024x1024xf32>,
                   %output: memref<1024x1024xf32>) {
  // Matrix-matrix multiplication.  
  linalg.matmul ins(%lhs, %rhs: memref<1024x1024xf32>, memref<1024x1024xf32>)
                outs(%output: memref<1024x1024xf32>)
  return
}

func.func @main() {
  %A = memref.alloc() : memref<1024x1024xf32>
  %B = memref.alloc() : memref<1024x1024xf32>
  %C = memref.alloc() : memref<1024x1024xf32>

  %cf1 = arith.constant 1.00000e+00 : f32

  linalg.fill ins(%cf1 : f32) outs(%A : memref<1024x1024xf32>)
  linalg.fill ins(%cf1 : f32) outs(%B : memref<1024x1024xf32>)

  %num_reps = arith.constant 10 : index
    
  linalg.fill ins(%cf1 : f32) outs(%C : memref<1024x1024xf32>)

  %t_start = call @rtclock() : () -> f64
  affine.for %arg0 = 0 to %num_reps {
    func.call @fc_relu(%A, %B, %C) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t = arith.subf %t_end, %t_start : f64

  %res = affine.load %C[0, 0]: memref<1024x1024xf32>
  vector.print %res: f32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %M = memref.dim %C, %c0 : memref<1024x1024xf32>
  %N = memref.dim %C, %c1 : memref<1024x1024xf32>
  %K = memref.dim %A, %c1 : memref<1024x1024xf32>

  // num_flops_per_iter = 2*M*N*K
  %f1 = arith.muli %M, %N : index
  %f2 = arith.muli %f1, %K : index
  %num_flops_per_iter = arith.muli %c2, %f2 : index

  // num_flops_total = num_flops_per_iter * num_reps
  %num_flops_total = arith.muli %num_flops_per_iter, %num_reps: index

  // Print the number of flops per second
  %num_flops_total_i = arith.index_cast %num_flops_total : index to i64
  %num_flops_total_f = arith.uitofp %num_flops_total_i : i64 to f64
  %flops_per_s = arith.divf %num_flops_total_f, %t : f64
  call @printFlops(%flops_per_s) : (f64) -> ()

  memref.dealloc %A : memref<1024x1024xf32>
  memref.dealloc %B : memref<1024x1024xf32>
  memref.dealloc %C : memref<1024x1024xf32>
  return
}

func.func private @printFlops(f64)
func.func private @rtclock() -> f64
