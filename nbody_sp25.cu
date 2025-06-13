// ---------------------------------------------------------------------------- 
// CUDA code to compute minimun distance between n points
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_POINTS 1048576

#define ERR_MALLOC 1
#define ERR_MEMCPY 2
#define ERR_KERNEL 3

// ----------------------------------------------------------------------------
// Device‐utility: atomic minimum on floats via compare‐and‐swap
__device__ float atomicMinFloat(float* address, float val) {
    int* addr_as_int = (int*)address;
    int  old_int     = *addr_as_int;
    float old_val    = __int_as_float(old_int);

    // only try update if new value is smaller
    while (val < old_val) {
        int new_int = __float_as_int(val);
        int prev_int = atomicCAS(addr_as_int, old_int, new_int);
        if (prev_int == old_int) {
            // we succeeded
            break;
        }
        // another thread updated it first—reload and retry if still smaller
        old_int  = prev_int;
        old_val  = __int_as_float(old_int);
    }
    return old_val;
}

// ---------------------------------------------------------------------------- 
// Kernel Function to compute distance between all pairs of points
// Input: 
//	X: X[i] = x-coordinate of the ith point
//	Y: Y[i] = y-coordinate of the ith point
//	n: number of points
// Output: 
//	D: D[0] = minimum distance
//
// -----------------------------------------------
// Optimized kernel: shared‐memory tiling + block‐level reduction
// Shared memory size must be (2 * blockDim.x * sizeof(float))
__global__ void minimum_distance(float *X, float *Y, float *D, int n) {
    extern __shared__ float sdata[]; 
    float *sx = sdata;                    // blockDim.x floats for X
    float *sy = &sdata[blockDim.x];       // blockDim.x floats for Y

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_min = INFINITY;

    // Load this thread’s point
    float xi = (i < n) ? X[i] : 0.0f;
    float yi = (i < n) ? Y[i] : 0.0f;

    // Loop over all “tiles” of size blockDim.x
    for (int tile = 0; tile * blockDim.x < n; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        // Cooperative load into shared memory
        if (idx < n) {
            sx[threadIdx.x] = X[idx];
            sy[threadIdx.x] = Y[idx];
        } else {
            sx[threadIdx.x] = 0.0f;
            sy[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute distances against this tile
        if (i < n) {
            for (int j = 0; j < blockDim.x; ++j) {
                int idx2 = tile * blockDim.x + j;
                if (idx2 > i && idx2 < n) {
                    float dx = xi - sx[j];
                    float dy = yi - sy[j];
                    float dist = sqrtf(dx*dx + dy*dy);
                    if (dist < local_min) local_min = dist;
                }
            }
        }
        __syncthreads();
    }

    // Block‐wide reduction to find the minimum of local_min across threads
    sdata[threadIdx.x] = local_min;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = sdata[threadIdx.x + offset];
            if (other < sdata[threadIdx.x]) {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically updates the global minimum
    if (threadIdx.x == 0) {
        atomicMinFloat(D, sdata[0]);
    }
}

// ---------------------------------------------------------------------------- 
// Host function to compute minimum distance between points
// Input:
//	X: X[i] = x-coordinate of the ith point
//	Y: Y[i] = y-coordinate of the ith point
//	n: number of points
// Output: 
//	D: minimum distance
//
float minimum_distance_host(float * X, float * Y, int n) {
    float dx, dy, Dij, min_distance, min_distance_i;
    int i, j;
    dx = X[1]-X[0];
    dy = Y[1]-Y[0];
    min_distance = sqrtf(dx*dx+dy*dy);
    for (i = 0; i < n-1; i++) {
	for (j = i+1; j < i+2; j++) {
	    dx = X[j]-X[i];
	    dy = Y[j]-Y[i];
	    min_distance_i = sqrtf(dx*dx+dy*dy);
	}
	for (j = i+1; j < n; j++) {
	    dx = X[j]-X[i];
	    dy = Y[j]-Y[i];
	    Dij = sqrtf(dx*dx+dy*dy);
	    if (min_distance_i > Dij) min_distance_i = Dij;
	}
	if (min_distance > min_distance_i) min_distance = min_distance_i;
    }
    return min_distance;
}

// ---------------------------------------------------------------------------- 
// Print device properties
void print_device_properties() {
    int i, deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for ( i = 0; i < deviceCount; ++i ) {
	cudaGetDeviceProperties(&deviceProp, i);
	printf("[Device: %1d] Compute Capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
	printf(" ... multiprocessor count  = %d\n", deviceProp.multiProcessorCount); 
	printf(" ... max threads per multiprocessor = %d\n", deviceProp.maxThreadsPerMultiProcessor); 
	printf(" ... max threads per block = %d\n", deviceProp.maxThreadsPerBlock); 
	printf(" ... max block dimension   = %d, %d, %d (along x, y, z)\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]); 
	printf(" ... max grid size         = %d, %d, %d (along x, y, z)\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]); 
	printf(" ... warp size             = %d\n", deviceProp.warpSize); 
	printf(" ... clock rate            = %d MHz\n", deviceProp.clockRate/1000); 
    }
    printf("------------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------- 
// CUDA errors
void check_error(cudaError_t err, int type) {
    if (err != cudaSuccess) {
        switch(type) {
            case ERR_MALLOC:
                fprintf(stderr, "Failed cudaMalloc (error code %s)!\n", cudaGetErrorString(err));
                break;
            case ERR_MEMCPY:
                fprintf(stderr, "Failed cudaMemcpy (error code %s)!\n", cudaGetErrorString(err));
                break;
            case ERR_KERNEL:
                fprintf(stderr, "Failed kernel launch (error code %s)!\n", cudaGetErrorString(err));
                break;
        }
        exit(0);
    }
}


// ---------------------------------------------------------------------------- 
// Main program - initializes points and computes minimum distance 
// between the points
//
int main(int argc, char* argv[]) {
    // Host Data
    float * hVx;            // host x-coordinate array
    float * hVy;            // host y-coordinate array
    float   hmin_dist;      // minimum value on host

    // Device Data
    float * dVx;            // device x-coordinate array
    float * dVy;            // device y-coordinate array
    float * dmin_dist;      // minimum value on device

    // Device parameters
    int    MAX_BLOCK_SIZE;      // Maximum number of threads allowed on the device
    int    blocks;              // Number of blocks in grid
    int    threads_per_block;   // Number of threads per block
    size_t shared_bytes;        // Shared memory size for kernel launch

    // Timing variables
    cudaEvent_t start, stop;                // GPU timing variables
    struct timespec cpu_start, cpu_stop;    // CPU timing variables
    float time_array[10];

    // Other variables
    cudaError_t err = cudaSuccess;
    int i, size, num_points;
    int seed = 0;

    // Print device properties
    print_device_properties();

    // Get device information and set device to use
    int deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);
        cudaGetDeviceProperties(&deviceProp, 0);
        MAX_BLOCK_SIZE = deviceProp.maxThreadsPerBlock;
    } else {
        printf("Warning: No GPU device found ... results may be incorrect\n");
    }

    // Timing initializations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Check input
    if (argc != 2) {
        printf("Use: %s <number of points>\n", argv[0]);
        exit(0);
    }
    num_points = atoi(argv[1]);
    if (num_points < 2) {
        printf("Minimum number of points allowed: 2\n");
        exit(0);
    }
    if (num_points > MAX_POINTS) {
        printf("Maximum number of points allowed: %d\n", MAX_POINTS);
        exit(0);
    }

    // Allocate host coordinate arrays
    size = num_points * sizeof(float);
    hVx = (float *) malloc(size);
    hVy = (float *) malloc(size);

    // Initialize points
    srand48(seed);
    float sqrtn = sqrtf((float)num_points);
    for (i = 0; i < num_points; i++) {
        hVx[i] = sqrtn * (float)drand48();
        hVy[i] = sqrtn * (float)drand48();
    }

    // Allocate device coordinate arrays
    err = cudaMalloc(&dVx,      size);      check_error(err, ERR_MALLOC);
    err = cudaMalloc(&dVy,      size);      check_error(err, ERR_MALLOC);
    // only need one float for the minimum, but allocate same size for simplicity
    err = cudaMalloc(&dmin_dist, size);     check_error(err, ERR_MALLOC);

    // Copy coordinate arrays from host to device
    cudaEventRecord(start, 0);
    err = cudaMemcpy(dVx, hVx, size, cudaMemcpyHostToDevice);  check_error(err, ERR_MEMCPY);
    err = cudaMemcpy(dVy, hVy, size, cudaMemcpyHostToDevice);  check_error(err, ERR_MEMCPY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_array[0], start, stop);

    // ------------------------------------------------------------
    // Prepare and launch optimized kernel
    // ------------------------------------------------------------

    // Initialize device minimum to +INFINITY
    {
        float hInf = INFINITY;
        err = cudaMemcpy(dmin_dist, &hInf, sizeof(float), cudaMemcpyHostToDevice);
        check_error(err, ERR_MEMCPY);
    }

    // Configure kernel launch parameters
    threads_per_block = (MAX_BLOCK_SIZE < 256 ? MAX_BLOCK_SIZE : 256);
    blocks             = (num_points + threads_per_block - 1) / threads_per_block;
    shared_bytes       = 2 * threads_per_block * sizeof(float);  // for sx[] and sy[]

    // Launch kernel and time only the execution
    cudaEventRecord(start, 0);
    minimum_distance<<<blocks, threads_per_block, shared_bytes>>>(dVx, dVy, dmin_dist, num_points);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_array[1], start, stop);

    // Check for kernel errors
    err = cudaGetLastError();  check_error(err, ERR_KERNEL);

    // Copy result back from device to host
    cudaEventRecord(start, 0);
    err = cudaMemcpy(&hmin_dist, dmin_dist, sizeof(float), cudaMemcpyDeviceToHost);
    check_error(err, ERR_MEMCPY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_array[2], start, stop);

    // Compute reference result on host
    clock_gettime(CLOCK_REALTIME, &cpu_start);
    float host_min = minimum_distance_host(hVx, hVy, num_points);
    clock_gettime(CLOCK_REALTIME, &cpu_stop);
    time_array[3] = 1000.0f * ((cpu_stop.tv_sec  - cpu_start.tv_sec)
                             + (cpu_stop.tv_nsec - cpu_start.tv_nsec) * 1e-9f);

    // Print results
    printf("Number of Points    = %d\n", num_points);
    printf("GPU Host-to-device  = %f ms\n", time_array[0]);
    printf("GPU execution time  = %f ms\n", time_array[1]);
    printf("GPU Device-to-host  = %f ms\n", time_array[2]);
    printf("CPU execution time  = %f ms\n", time_array[3]);
    printf("Min. distance (GPU) = %e\n", hmin_dist);
    printf("Min. distance (CPU) = %e\n", host_min);
    printf("Relative error      = %e\n", fabs(host_min - hmin_dist) / host_min);

    // Cleanup
    cudaFree(dVx);
    cudaFree(dVy);
    cudaFree(dmin_dist);
    free(hVx);
    free(hVy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
