#include <stdio.h>
#include <cuda_runtime.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

__global__ void benchmark_coo_spmv(int nonzero, int* rows, int* cols, float* vals, float* x, float* y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //int num_nonzeros = nonzero;

    if (idx < nonzero) {
        // warmup    
        y[rows[idx]] += vals[idx] * x[cols[idx]];

        // 500 Iterations
        int num_iterations;
        num_iterations = 500;

        //printf("\tPerforming %d iterations\n", num_iterations);
        
        for(int j = 0; j < num_iterations; j++) {
            y[rows[idx]] += vals[idx] * x[cols[idx]];
        }

        //printf("I am %d!\n", idx);
    }
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult, especially in single precision
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
        // coo.vals[i] = 1.0;
    }
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

#ifdef TESTING
//print in COO format
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COOcu", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif 

    //initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));

    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
        // x[i] = 1;
    }

    //printf("38th value x %f\n", x[37]);

    for(int i = 0; i < coo.num_rows; i++)
        y[i] = 0;

    // printf("38th value y %f\n", y[37]);
    // printf("Bytes %d vs bytes 2 %d vs bytes 3 %d\n", coo.num_cols*sizeof(float), sizeof(x), sizeof(y));

    float* yGPU = NULL;
    float* xGPU = NULL;
    int* colsGPU = NULL;
    int* rowsGPU = NULL;
    float* valsGPU = NULL;

    // Allocate space on GPU for copies of the data
    if (cudaMalloc((void**)&yGPU, coo.num_rows*sizeof(float)) != cudaSuccess) {
        printf("fail 1\n");
        exit(1);
    }

    if (cudaMalloc((void**)&xGPU, coo.num_cols*sizeof(float)) != cudaSuccess) {
        printf("fail 2\n");
        exit(1);
    }
    
    if (cudaMalloc((void**)&colsGPU, coo.num_nonzeros*sizeof(int)) != cudaSuccess) {
        printf("fail 3\n");
        exit(1);
    }

    if (cudaMalloc((void**)&rowsGPU, coo.num_nonzeros*sizeof(int)) != cudaSuccess) {
        printf("fail 4\n");
        exit(1);
    }

    if (cudaMalloc((void**)&valsGPU, coo.num_nonzeros*sizeof(float)) != cudaSuccess) {
        printf("fail 5\n");
        exit(1);
    }

    // Copy Sequences over to the device
    if (cudaMemcpy(yGPU, y, coo.num_rows*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("fail 6\n");
        exit(1);
    }

    if (cudaMemcpy(xGPU, x, coo.num_cols*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("fail 7\n");
        exit(1);
    }

    if (cudaMemcpy(colsGPU, coo.cols, coo.num_nonzeros*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("fail 8\n");
        exit(1);
    }

    if (cudaMemcpy(rowsGPU, coo.rows, coo.num_nonzeros*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("fail 9\n");
        exit(1);
    }

    if (cudaMemcpy(valsGPU, coo.vals, coo.num_nonzeros*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("fail 10\n");
        exit(1);
    }

    /* Benchmarking */
    // Block and grid dimensions.
    int threadsPerBlock = 100;
    // Round up for the number of blocks we need.
    int blocksPerGrid = ( coo.num_nonzeros + threadsPerBlock - 1 ) / threadsPerBlock;

    // 200 Iterations
    int num_iterations;
    num_iterations = 500;

    timer t;
    timer_start(&t);

    benchmark_coo_spmv<<<blocksPerGrid, threadsPerBlock>>>(coo.num_nonzeros, rowsGPU, colsGPU, valsGPU, xGPU, yGPU);
    if (cudaGetLastError() != cudaSuccess) {
        printf("fail 11\n");
        exit(1);
    }
    
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    //double sec_per_iteration = msec_per_iteration / 1000.0;
    //double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    //double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.10f ms\n", msec_per_iteration); 

    // Copy Back Results

    float * hostx = (float*)malloc(coo.num_cols * sizeof(float));
    float * hosty = (float*)malloc(coo.num_rows * sizeof(float));
    // printf("Col num %d\n", coo.num_cols);
    // printf("Bytes %d vs bytes 2 %d vs bytes 3 %d\n", coo.num_cols*sizeof(float), sizeof(x[0])*coo.num_cols, sizeof(hostx[0])*coo.num_cols);
    int err = cudaMemcpy(hostx, xGPU, coo.num_cols*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("%d fail 12\n", err);
        exit(1);
    }

    if (cudaMemcpy(hosty, yGPU, coo.num_rows*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("fail 13\n");
        exit(1);
    }

    // Free CUDA Memory
    cudaFree(xGPU);
    cudaFree(yGPU);
    cudaFree(colsGPU);
    cudaFree(rowsGPU);
    cudaFree(valsGPU);

    cudaDeviceReset();

    // double coo_gflops;
    // coo_gflops = benchmark_coo_spmv(&coo, x, y);

    /* Test correctnesss */
#ifdef TESTING
    printf("Writing x and y vectors ...");
    fp = fopen("test_xcu2", "w");
    for (int i=0; i<coo.num_cols; i++)
    {
      fprintf(fp, "%f\n", hostx[i]);
    }
    fclose(fp);
    fp = fopen("test_ycu2", "w");
    for (int i=0; i<coo.num_rows; i++)
    {
      fprintf(fp, "%f\n", hosty[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif

    delete_coo_matrix(&coo);
    free(hostx);
    free(hosty);
    free(x);
    free(y);

    return 0;
}

