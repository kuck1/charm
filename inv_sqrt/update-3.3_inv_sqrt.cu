#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define DGEMM dgemm_
#define DSPEV dspev_
#define PRINTF printf
#define EXIT exit
#define CLOCKS_PER_SEC_C  1000000
#define MAXTIME 2147.48

#define MAX_BLOCKS 65521          // Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
#define THREADS_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void tolerance_check(double *tol_now, double *d_Titer, double *d_mat3, int nstate_sq);

void cputime(double *);
void get_iter_Tmat(double *,double *,int );
void get_diag_Tmat(double *,double *,int );
void get_unit_Tmat(double *,int );

extern "C" { void DGEMM (char *, char *, int *, int *, int *,double *,double *, int *, double *, int *, double *, double *, int * ); }

void matmul(double *X, int *LDX, int *ITYPE_X, 
    double *Y, int *LDY, int *ITYPE_Y,
    double *Z, int *LDZ, int *NRZ, int *NCZ, int *NXY,
    double *ALPHA, double *BETA)
{
    int m = *NRZ;
    int n = *NCZ;
    int k = *NXY;

    //char MATX=(ITYPE_X) ? 'N' : 'T';
    //char MATY=(ITYPE_Y) ? 'N' : 'T';
    // DGEMM(&MATX,&MATY,NRZ,NCZ,NXY,ALPHA,X,LDX,Y,LDY,BETA,Z,LDZ);

    cublasOperation_t MATX = (ITYPE_X) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t MATY = (ITYPE_Y) ? CUBLAS_OP_N : CUBLAS_OP_T;

    // cudaError_t cudaStat;           // cudaMalloc status 
    // cublasStatus_t stat;            // CUBLAS functions status 
    cublasHandle_t handle;          // CUBLAS context

    // Step 1: Allocate memory on the device:
    double *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, (m*k)*sizeof(double));          // X is an m x k matrix
    cudaMalloc(&d_Y, (k*n)*sizeof(double));          // Y is a  k X n matix
    cudaMalloc(&d_Z, (m*n)*sizeof(double));          // Z is an m x n matix

    cublasCreate(&handle);                           // initialize CUBLAS context

    // Step 2: Initailize device memory from host:
    cublasSetMatrix(m, k, sizeof(double), X, m, d_X, m);       
    cublasSetMatrix(k, n, sizeof(double), Y, k, d_Y, k); 
    cublasSetMatrix(m, n, sizeof(double), Z, m, d_Z, m);

    // Step 3: Perform operation, function launches kernel on GPU itself
    cublasDgemm(handle, MATX, MATY, m, n, k, ALPHA, d_X, m, d_Y, k, BETA, d_Z, m);

    // Step 4: Copy the result back to the host:
    cublasGetMatrix(m, n, sizeof(double), d_Z, m, Z, m);

    // Step 5: Clean up
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
    cublasDestroy(handle);
}

//DGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )

/* cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
                              int m, int n, int k, const double *alpha, const double *A, int lda, 
                              const double *B, int ldb, const double *beta, double *C, int ldc)
*/

void device_matmul(double *d_X, int *LDX, int *ITYPE_X, 
    double *d_Y, int *LDY, int *ITYPE_Y,
    double *d_Z, int *LDZ, int *NRZ, int *NCZ, int *NXY,
    double *ALPHA, double *BETA, cublasHandle_t handle)
{
    int m = *NRZ;
    int n = *NCZ;
    int k = *NXY;

    cublasOperation_t MATX = (ITYPE_X) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t MATY = (ITYPE_Y) ? CUBLAS_OP_N : CUBLAS_OP_T;

    //cublasHandle_t handle;          // CUBLAS context

    cublasDgemm(handle, MATX, MATY, m, n, k, ALPHA, d_X, m, d_Y, k, BETA, d_Z, m);
}


#define _USE_LAPACK_

#ifdef _USE_LAPACK_
extern "C" {void   DSPEV(char *, char *, int *, double [], double [], double [], int *, double [], int *);}
#endif
//=======================================================================
//ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//=======================================================================
  int main()
//=======================================================================
  {// begin routine 
//=======================================================================
// I) Set up the problem

   int nstate;
   PRINTF("\n============================================\n");
   PRINTF("Enter the matrix size : ");
   if(scanf("%d",&nstate)){}
   else{printf("%s\n", "Please enter an integer");}
   int nstate_sq = nstate*nstate;

   double *S     = new double[nstate_sq];
   double *Tunit = new double[nstate_sq];
   double *Tdiag = new double[nstate_sq];
   double *Titer = new double[nstate_sq];
   
   PRINTF("Using random input\n\n");
   for(int i=0;i<nstate_sq;i++){S[i]=0.0;}
   for(int i=0;i<nstate;i++){int ind =i+nstate*i;S[ind]=2.0;}

   double seed=14571.0;
   srand48((long) seed);
   for(int i=0;i<nstate;i++){
   for(int j=i;i<nstate;i++){
     int ind  = i+nstate*j;
     int indt = j+nstate*i;
     //int ierr=0; // n=1; -------- *** Not used ?
     double rand=drand48();
     S[ind]  += (rand-0.5)*2.0e-3;
     S[indt] = S[ind];
   }}//endfor

//=======================================================================
// II) Try three methods
   
   get_unit_Tmat(Tunit,nstate);
   get_diag_Tmat(S,Tdiag,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);
   get_iter_Tmat(S,Titer,nstate);

//=======================================================================
// III) Check the error of the iterative method

   double err=0.0;
   for(int i=0;i<nstate_sq;i++){
     double tmp=Tdiag[i]-Titer[i];    
     tmp = tmp*tmp;
     err = (err > tmp ? err : tmp);
   }//endfor
   err = sqrt(err);
   PRINTF("Maximum error in any element : %g\n",err);

   err=0.0;
   for(int i=0;i<nstate;i++){
   for(int j=i;j<nstate;j++){
     int ind  = i + j*nstate;
     int indt = j + i*nstate;
     double tmp=Titer[ind]-Titer[indt];    
     tmp = tmp*tmp;
     err = (err > tmp ? err : tmp);
   }}//endfor
   err = sqrt(err);
   PRINTF("Deviation from symmetric : %g\n",err);
   PRINTF("============================================\n\n");

//=======================================================================
  }//end routine
//=======================================================================



//============================================================================
//cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//============================================================================
// Diagonalize S and construct T=S^{-1/2} using eigenvalues and eigenvectors
//============================================================================

void get_diag_Tmat(double *S,double *T,int nstate)

//============================================================================
  {//begin routine
//============================================================================
// I) Get some scratch

   double cpu1,cpu2;
   cputime(&cpu1);

   int nstate_sq     = nstate*nstate;
   double *umat      = new double[nstate_sq];
   double *scr_mat1  = new double[nstate_sq];
   double *scr_mat2  = new double[nstate_sq];
   double *s_eigs    = new double[nstate];
   double *scr1      = new double[3*nstate];        
   double *scr2      = new double[3*nstate];


//==========================================================================
// II. Diagonalize S using rs_ FORTRAN diagonalization routine

  int ifound = 0;
  int ierr   = 0;

  //----------------------------------------------------------------------
  // Use LAPACK : Captain Jack is Happy.
#ifdef _USE_LAPACK_
   ifound ++;
   for(int i = 1; i <= nstate; i++){
   for(int j = 1; j <= i; j++){
     int ind  = (i-1) + (j-1)*nstate;
     int ind2 = (i-1) + (j-1)*(2*nstate-j)/2;
     scr_mat1[ind2] = S[ind];
   }}//endfor
   char Vstuff    ='V';
   char Lstuff    ='L';
   DSPEV(&Vstuff,&Lstuff,&nstate,scr_mat1,s_eigs,umat,&nstate,scr1,&ierr);
#endif


   if(ifound!=1 || ierr != 0){
     PRINTF("@@@@@@@@@@@@@@@@@@@@_error_@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
     PRINTF("Error trying to diagonalize S : %d %d\n",ifound,ierr);
     PRINTF("@@@@@@@@@@@@@@@@@@@@_error_@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
     EXIT(1);
   }//endif

//==========================================================================
// III. Compute inverse square root of eigenvalues:  Occupation numbers 
//      are HACKED!!!!!

  //----------------------------------------------------------------------
  // A) Construct diagonal matrix using eigenvalues : sqrt(2/lamba)

   for(int i = 0; i < nstate; i++){s_eigs[i] = sqrt(2.0/s_eigs[i]);}
   memset(scr_mat1,0,sizeof(double)*nstate_sq);
   for(int i = 0; i < nstate; i++){
     int ind = i*nstate+i;
     scr_mat1[ind]=s_eigs[i];
   }/* endfor */

  //------------------------------------------------------------------------
  // B) Transform matrix back to original representation using eigenvectors

   double alpha = 1.0; double beta = 0.0;
   int itransp  = 0;   int inorm   = 1;

   matmul(scr_mat1,&nstate,&inorm,umat,&nstate,&itransp,scr_mat2,
             &nstate,&nstate,&nstate,&nstate,&alpha,&beta);
   matmul(umat,&nstate,&inorm,scr_mat2,&nstate,&inorm,T,
             &nstate,&nstate,&nstate,&nstate,&alpha,&beta);


//============================================================================
// IV) Free allocated temporary memory

   delete [] umat;
   delete [] scr_mat1;
   delete [] scr_mat2;
   delete [] s_eigs;
   delete [] scr1;
   delete [] scr2;

   cputime(&cpu2);
   PRINTF("nstate %d : cpu time diag : %g\n\n",nstate,cpu2-cpu1);

//============================================================================
  } /* End function */
//============================================================================



//============================================================================
//cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//============================================================================
// Set Tmax to the Unit matrix : remove cputime overhead of diag to test
//                               parallel performance
//============================================================================
void get_unit_Tmat(double *Tunit,int nstate){
   int nstate_sq = nstate*nstate;
   memset(Tunit,0,nstate_sq*sizeof(double));
   for(int i=0;i<nstate;i++){int ind = i+i*nstate;Tunit[ind] = 1.0;}
}
//============================================================================


/*==========================================================================*/
/*cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc*/
/*==========================================================================*/
/* Kernel to check tolerance directly on device */
/*==========================================================================*/
__global__
void tolerance_check(double *d_tol_now, double *d_Titer, double *d_mat3, int nstate_sq)

//============================================================================
{//begin routine
//============================================================================
// I) Find location

    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int local_idx = threadIdx.x;

// II) d_mat3 = (d_mat3 - d_Titer) ^ 2
    if(global_idx < nstate_sq){                                 // Needed in case last block is not full
        double tmp = d_mat3[global_idx] - d_Titer[global_idx];
        d_mat3[global_idx] = tmp * tmp;
    }
    __syncthreads();  // Every thread will execute this unconditionally

// III) Reduce any sized array
    if(global_idx < nstate_sq){                                 // Needed in case last block is not full
        unsigned int length = THREADS_PER_BLOCK, next_length, block2_size, global_start_idx, global_split_idx = 0, global_end_idx;

        if((blockIdx.x + 1) == gridDim.x) {   // This is the last block - may not be full
            length = nstate_sq - THREADS_PER_BLOCK * (gridDim.x - 1); // THREADS_PER_BLOCK == blockDim.x
        }
       
        for( ; length != 1; length = next_length){   // Ultimately we want host to do ~ 2 ^ 13 work, here it does 2^10
            next_length = (length + 1) / 2;
            global_start_idx = blockDim.x * blockIdx.x;
            global_split_idx = global_start_idx + next_length;
            global_end_idx = global_start_idx + length;
            block2_size = length / 2;    //length - next_length;
            if(global_split_idx <= global_idx && global_idx < global_end_idx){ 
                d_mat3[global_idx - block2_size] += d_mat3[global_idx];
            } 
            __syncthreads();   // FIX : find way to have every thread execute this unconditionally
        }

// IV) Only 0 thread from each block copies result back

        if(local_idx == 0){
            d_tol_now[blockIdx.x] = d_mat3[global_idx];
        }
    }
}
/*==========================================================================*/


//============================================================================
//cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//============================================================================
// Schulz iteration for inverse sqrt root : quadratic convergence!
//============================================================================

void get_iter_Tmat(double *S,double *Titer,int nstate)

//============================================================================
{//begin routine
//============================================================================
// I) Get some scratch on the host

   double cpu1,cpu2; 
   cputime(&cpu1);

   int nstate_sq     = nstate*nstate;
   double *tol_now_ptr = new double[MAX_BLOCKS];
   double *scr_mat1  = new double[nstate_sq];
   double *scr_mat2  = new double[nstate_sq];
   double *scr_mat3  = new double[nstate_sq];

//============================================================================
// II) Set up CUBLAS context

    // cudaError_t cudaStat;           // cudaMalloc status 
    // cublasStatus_t stat;            // CUBLAS functions status 
    cublasHandle_t handle;              // CUBLAS context
//============================================================================
// III) Allocate memory on the device
    
    double *d_Titer, *d_mat1, *d_mat2, *d_mat3, *d_tol_now;
    cudaMalloc(&d_Titer, nstate_sq*sizeof(double));
    cudaMalloc(&d_mat1, nstate_sq*sizeof(double));          
    cudaMalloc(&d_mat2, nstate_sq*sizeof(double));          
    cudaMalloc(&d_mat3, nstate_sq*sizeof(double));        
    cudaMalloc(&d_tol_now, MAX_BLOCKS*sizeof(double)); 
    cublasCreate(&handle);                           // initialize CUBLAS context
//============================================================================
// IV) Schulz iteration

  //--------------------------------------------------------------------
  // A) Initialize scr_mat1 and Titer on host

    // scr_mat1 = S/2
    for(int i=0;i<nstate_sq;i++){scr_mat1[i] = S[i]/2.0;}
    // Titer = I = unit matrix
    memset(Titer,0,nstate_sq*sizeof(double));
    for(int i=0;i<nstate;i++){int ind = i+i*nstate;Titer[ind] = 1.0;}

    //--------------------------------------------------------------------
    // B) Initailize d_mat1 and d_Titer on device
    cublasSetMatrix(nstate, nstate, sizeof(double), scr_mat1, nstate, d_mat1, nstate);       
    cublasSetMatrix(nstate, nstate, sizeof(double), Titer, nstate, d_Titer, nstate);

    //--------------------------------------------------------------------
    // C) Iterate

    int iter        = 0;
    double tol_now  = 1.0;
    while (tol_now > 1.0e-15 && iter<10){
        iter++;
        //--------------------------------
        // scr_mat2 =  3*I - Titer*scr_mat1 
        int itransp  = 0;    int inorm    = 1;
        double alpha = -1.0; double beta  = 1.0;
        memset(scr_mat2,0,nstate_sq*sizeof(double));
        for(int i=0;i<nstate;i++){int ind = i+i*nstate;scr_mat2[ind]=3.0;}
        cublasSetMatrix(nstate, nstate, sizeof(double), scr_mat2, nstate, d_mat2, nstate);
        device_matmul(d_Titer,&nstate,&inorm,d_mat1,&nstate,&itransp,d_mat2,
            &nstate,&nstate,&nstate,&nstate,&alpha,&beta,handle);
        //--------------------------------
        // scr_mat1 = 0.5*scr_mat1*scr_mat2 = 0.5*scr_mat3*scr_mat2
        alpha = 0.5;  beta  = 0.0;
        cudaMemcpy(d_mat3,d_mat1,nstate_sq*sizeof(double),cudaMemcpyDeviceToDevice);
        device_matmul(d_mat3,&nstate,&inorm,d_mat2,&nstate,&itransp,d_mat1,
            &nstate,&nstate,&nstate,&nstate,&alpha,&beta,handle);
        //--------------------------------
        // Titer = 0.5*scr_mat2*Titer = 0.5*scr_mat2*scr_mat3
        alpha = 0.5;  beta  = 0.0;
        cudaMemcpy(d_mat3,d_Titer,nstate_sq*sizeof(double),cudaMemcpyDeviceToDevice);
        device_matmul(d_mat2,&nstate,&inorm,d_mat3,&nstate,&itransp,d_Titer,
            &nstate,&nstate,&nstate,&nstate,&alpha,&beta,handle);
        //--------------------------------
        // Launch kernel to check tolerance
        int grid_size = (nstate_sq + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
        tolerance_check<<<grid_size, THREADS_PER_BLOCK>>>(d_tol_now, d_Titer, d_mat3, nstate_sq);

        cudaMemcpy(tol_now_ptr, d_tol_now, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        tol_now = 0.0;
        for(int i = 0; i < grid_size; i ++){
            tol_now += tol_now_ptr[i];
        }
        tol_now = sqrt(tol_now / nstate_sq);  
        PRINTF("iter %d : tol %g\n",iter,tol_now);   
    }//endwhile

    if(tol_now>1.0e-15){
        PRINTF("@@@@@@@@@@@@@@@@@@@@_error_@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        PRINTF("Iterative computation of S^{-1/2} failed\n");
        PRINTF("@@@@@@@@@@@@@@@@@@@@_error_@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        EXIT(1);
    }//endif

/*==========================================================================*/
// V) Copy the result back to the host
    cublasGetMatrix(nstate, nstate, sizeof(double), d_Titer, nstate, Titer, nstate);

/*==========================================================================*/
// VI) Clean up device
    cudaFree(d_Titer);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
    cudaFree(d_tol_now);
    cublasDestroy(handle);

// VII) Clean up host

    delete [] tol_now_ptr;
    delete [] scr_mat1;
    delete [] scr_mat2;
    delete [] scr_mat3;

    cputime(&cpu2);
    PRINTF("nstate %d : cpu time iter : %g\n\n",nstate,cpu2-cpu1);
/*==========================================================================*/
}//end routine
/*==========================================================================*/


/*==========================================================================*/
/*cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc*/
/*==========================================================================*/
/* subroutine to time processes */
/*==========================================================================*/

void cputime(double *time)

/*==========================================================================*/
{
  int itime;
  static double to=0.,tn=0.;

  itime = clock();
  tn = (double)((double)itime/(double)CLOCKS_PER_SEC_C);
  *time = tn;
  if(tn >= 0 && to >= 0){*time=tn;}
  if(tn < 0  && to >= 0){*time=MAXTIME*2.0+tn;}
  if(tn >= 0 && to <  0){*time=tn+MAXTIME;}
  if(tn <  0 && to <  0){*time=MAXTIME+tn;}

  to = tn;
}
/*==========================================================================*/
