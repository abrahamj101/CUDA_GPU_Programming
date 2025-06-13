# CSCE 435 - Homework 6: CUDA Minimum Distance Optimization

This project enhances an N-body CUDA program to efficiently compute the minimum distance between points using advanced parallel techniques. The assignment focuses on improving memory access, reducing atomic contention, and optimizing kernel performance.

---

## 📁 File Modified
- `nbodu_sp25.cu`

---

## 🔧 Code Enhancements

### 1. `atomicMinFloat`: Atomic Minimum Helper for Floats
- Introduced a `__device__` function using `atomicCAS` on float bit-patterns.
- **Why:** CUDA does not natively support `atomicMin` for floats. This helper enables safe, thread-level updates to a global float minimum.

---

### 2. Optimized CUDA Kernel: `minimum_distance`

#### Improvements:
- **Shared Memory Tiling**  
  Threads cooperatively load X and Y values into `__shared__` arrays, minimizing global memory traffic.
  
- **Per-Thread Local Minima**  
  Each thread calculates a local minimum within its tile.

- **Block-Level Reduction**  
  Local minima are reduced within each block into a shared block-level minimum.

- **Single Atomic per Block**  
  Only thread 0 performs an `atomicMinFloat()` once per block, reducing atomic operation overhead.

---

### 3. Host-Side Initialization and Kernel Launch

- **Min Value Initialization**  
  Initialized `dmin_dist` with `+INFINITY` before launch to ensure valid comparisons.

- **Dynamic Kernel Launch Parameters**  
  Used `deviceProp.maxThreadsPerBlock` to determine optimal:
  - `threads_per_block` (capped at 256)
  - `blocks = ceil(n / threads_per_block)`
  - `shared_bytes = 2 * threads_per_block * sizeof(float)`

- **Kernel Call**  
  ```cpp
  minimum_distance<<<blocks, threads_per_block, shared_bytes>>>(
      dVx, dVy, dmin_dist, num_points
  );

## 🔍 4. Robust Error Checking & Timing

To ensure correctness and measure efficiency, several runtime diagnostics were added:

### ✅ Error Handling
- All CUDA API calls (`cudaMalloc`, `cudaMemcpy`, kernel launches) are now wrapped with a helper function `check_error()`.
- This allows for immediate detection of failures such as memory allocation issues or kernel faults.

### ⏱️ Event-Based Timing
- CUDA events are used to track:
  - **Host-to-Device (H2D) transfer time**
  - **Kernel execution time**
  - **Device-to-Host (D2H) transfer time**
- **CPU-only execution time** is also preserved as a reference baseline for speedup comparison.

These metrics help validate performance gains achieved by the CUDA version relative to the original CPU implementation.

---

## 📐 5. Function Ordering & Visibility

### `atomicMinFloat` Positioning
- The custom `atomicMinFloat()` device function was moved **above** the kernel definition to avoid compiler visibility issues.
- This ensures the function is recognized during kernel compilation.

### CPU Reference Function
- The function `minimum_distance_host()` was kept intact to:
  - Validate GPU output correctness
  - Serve as a serial baseline for speed comparison

---

## 📊 GPU vs CPU Performance Benchmark

### ❓ Question:
> For what value of `n` does the GPU code become faster than the CPU code?

### ✅ Answer:
> Starting from **n = 1024** (i.e., `k = 10`), the CUDA GPU implementation becomes **faster** than the CPU version.

This demonstrates the point at which parallel overhead is outweighed by GPU throughput, validating the optimization strategy.

---

## ✅ Summary

The additions in Parts 4 and 5 ensure:
- Reliable CUDA diagnostics
- Accurate timing and benchmarking
- Functional clarity in code structure
- A clear point of performance crossover between CPU and GPU

