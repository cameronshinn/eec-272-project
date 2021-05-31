#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <string>

#include "load_smtx.h"

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printCsr(
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const char* name)
{
    const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE)? 0:1 ;
    printf("matrix %s is %d-by-%d, nnz=%d, base=%d, output base-1\n", name, m, n, nnz, base);
    for(int row = 0 ; row < m ; row++){
        const int start = csrRowPtrA[row  ] - base;
        const int end   = csrRowPtrA[row+1] - base;
        for(int colidx = start ; colidx < end ; colidx++){
            const int col = csrColIndA[colidx] - base;
            const float Areg = csrValA[colidx];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

template <typename value_t>
std::vector<value_t> csr_to_dense(CSR<value_t> sparse) {
    std::vector<value_t> dense(sparse.nrows * sparse.ncols);  // Populate with zeros

    // Fill in non-zeros
    for (idx_t m = 0; m < sparse.nrows; m++) {
        offset_t row_start = sparse.row_ptrs[m];
        offset_t row_end = sparse.row_ptrs[m + 1];

        for (unsigned int i = row_start; i < row_end; i++) {
            idx_t n = sparse.col_idxs[i];
            value_t val = sparse.values[i];

            dense[m * sparse.ncols + n] = val;
        }
    }

    return dense;
}

int main(int argc, char*argv[])
{
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrC = NULL;
    pruneInfo_t info = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;

    // Read in pruned matrix and convert to dense
    CSR<float> A_sparse;
    load_smtx(argv[1], A_sparse);
    std::vector<float> A = csr_to_dense(A_sparse);

    int m = A_sparse.nrows;
    int n = A_sparse.ncols;
    const int lda = m;
    int* csrRowPtrC = NULL;
    int* csrColIndC = NULL;
    float* csrValC  = NULL;
    float *d_A = NULL;
    int *d_csrRowPtrC = NULL;
    int *d_csrColIndC = NULL;
    float *d_csrValC = NULL;
    size_t lworkInBytes = 0;
    char *d_work = NULL;
    int nnzC = 0;
    float percentage = std::stof(argv[2]);
    printf("example of pruneDense2csrByPercentage \n");
    printf("prune out %.1f percentage of A \n", percentage);
    printMatrix(m, n, A.data(), lda, "A");
/* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);
    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    status = cusparseCreatePruneInfo(&info);
    assert(CUSPARSE_STATUS_SUCCESS == status);
/* step 2: configuration of matrix C */
    status = cusparseCreateMatDescr(&descrC);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL );
    cudaStat1 = cudaMalloc ((void**)&d_A         , sizeof(float)*lda*n );
    cudaStat2 = cudaMalloc ((void**)&d_csrRowPtrC, sizeof(int)*(m+1) );
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    cudaStat1 = cudaMemcpy(d_A, A.data(), sizeof(float)*lda*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    /* step 3: query workspace */
    status = cusparseSpruneDense2csrByPercentage_bufferSizeExt(
        handle,
        m,
        n,
        d_A,
        lda,
        percentage,
        descrC,
        d_csrValC,
        d_csrRowPtrC,
        d_csrColIndC,
        info,
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    printf("lworkInBytes = %lld \n", (long long)lworkInBytes);
    if (NULL != d_work) { cudaFree(d_work); }
    cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
    assert(cudaSuccess == cudaStat1);
/* step 4: compute csrRowPtrC and nnzC */
    status = cusparseSpruneDense2csrNnzByPercentage(
        handle,
        m,
        n,
        d_A,
        lda,
        percentage,
        descrC,
        d_csrRowPtrC,
        &nnzC, /* host */
        info,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    printf("nnzC = %d\n", nnzC);
    if (0 == nnzC ){
        printf("C is empty \n");
        return 0;
    }
/* step 5: compute csrColIndC and csrValC */
    cudaStat1 = cudaMalloc ((void**)&d_csrColIndC, sizeof(int  ) * nnzC );
    cudaStat2 = cudaMalloc ((void**)&d_csrValC   , sizeof(float) * nnzC );
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    status = cusparseSpruneDense2csrByPercentage(
        handle,
        m,
        n,
        d_A,
        lda,
        percentage,
        descrC,
        d_csrValC,
        d_csrRowPtrC,
        d_csrColIndC,
        info,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
/* step 7: output C */
    csrRowPtrC = (int*  )malloc(sizeof(int  )*(m+1));
    csrColIndC = (int*  )malloc(sizeof(int  )*nnzC);
    csrValC    = (float*)malloc(sizeof(float)*nnzC);
    assert( NULL != csrRowPtrC);
    assert( NULL != csrColIndC);
    assert( NULL != csrValC);
    cudaStat1 = cudaMemcpy(csrRowPtrC, d_csrRowPtrC, sizeof(int  )*(m+1), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(csrColIndC, d_csrColIndC, sizeof(int  )*nnzC , cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(csrValC   , d_csrValC   , sizeof(float)*nnzC , cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");
/* free resources */
    if (d_A         ) cudaFree(d_A);
    if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
    if (d_csrColIndC) cudaFree(d_csrColIndC);
    if (d_csrValC   ) cudaFree(d_csrValC);
    if (csrRowPtrC  ) free(csrRowPtrC);
    if (csrColIndC  ) free(csrColIndC);
    if (csrValC     ) free(csrValC);
    if (handle      ) cusparseDestroy(handle);
    if (stream      ) cudaStreamDestroy(stream);
    if (descrC      ) cusparseDestroyMatDescr(descrC);
    if (info        ) cusparseDestroyPruneInfo(info);
    cudaDeviceReset();
    return 0;
}
