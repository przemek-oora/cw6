#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"
#include <sched.h>
#include <assert.h>
#include <math.h>

#define INDEX 100
#define SIZE 512

#define IDX(i, j, n) (((j)+ (i)*(n)))

int
chol(double *A, unsigned int n)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;
    
    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) {
            for (k = 0; k < j; ++k) {
                A[IDX(i, j, n)] -= A[IDX(i, k, n)] *
                A[IDX(j, k, n)];
            }
        }
        
        if (A[IDX(j, j, n)] < 0.0) {
            return (1);
        }
        
        A[IDX(j, j, n)] = sqrt(A[IDX(j, j, n)]);
        for (i = j + 1; i < n; i++)
            A[IDX(i, j, n)] /= A[IDX(j, j, n)];
    }
    
    return (0);
}

int
chol1(double *A, unsigned int n)
{
    register unsigned int i; //OPT 1
    register unsigned int j;
    register unsigned int k;
    
    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) {
            for (k = 0; k < j; ++k) {
                A[IDX(i, j, n)] -= A[IDX(i, k, n)] *
                A[IDX(j, k, n)];
            }
        }
        
        if (A[IDX(j, j, n)] < 0.0) {
            return (1);
        }
        
        A[IDX(j, j, n)] = sqrt(A[IDX(j, j, n)]);
        for (i = j + 1; i < n; i++)
            A[IDX(i, j, n)] /= A[IDX(j, j, n)];
    }
    
    return (0);
}

int
chol2(double *A, unsigned int n)
{
    register unsigned int i;
    register unsigned int j;
    register unsigned int k;
    register unsigned int local_size=n; //OPT 2
    
    for (j = 0; j < local_size; j++) {
        for (i = j; i < local_size; i++) {
            for (k = 0; k < j; ++k) {
                A[IDX(i, j, local_size)] -= A[IDX(i, k, local_size)] *
                A[IDX(j, k, local_size)];
            }
        }
        
        if (A[IDX(j, j, local_size)] < 0.0) {
            return (1);
        }
        
        A[IDX(j, j, local_size)] = sqrt(A[IDX(j, j, local_size)]);
        for (i = j + 1; i < local_size; i++)
            A[IDX(i, j, local_size)] /= A[IDX(j, j, local_size)];
    }
    
    return (0);
}

int
chol3(double *A, unsigned int n)
{
    register unsigned int i;
    register unsigned int j;
    register unsigned int k;
    register unsigned int local_size=n; //OPT 2
    
    for (j = 0; j < local_size; j++) {
        for (i = j; i < local_size; i++) {
            for (k = 0; k < j; ++k) {
                A[IDX(i, j, local_size)] -= A[IDX(i, k, local_size)] *
                A[IDX(j, k, local_size)];
            }
        }
        
        if (A[IDX(j, j, local_size)] < 0.0) {
            return (1);
        }
        
        A[IDX(j, j, local_size)] = sqrt(A[IDX(j, j, local_size)]);
        for (i = j + 1; i < local_size; i++)
            A[IDX(i, j, local_size)] /= A[IDX(j, j, local_size)];
    }
    
    return (0);
}



static void test_fail(char *file, int line, char *call, int retval);

int main(int argc, char **argv) {
  extern void dummy(void *);
  float matrixa[INDEX][INDEX], matrixb[INDEX][INDEX], mresult[INDEX][INDEX];
  float real_time, proc_time, mflops;
  long long flpins;
  int retval;
  int i,j,k,l;

long long fp_ops[9];
long long ld_ins[9];

int events[2] = {PAPI_FP_OPS, PAPI_LD_INS};
long long values[2] = {0,};
int eventSet = PAPI_NULL;
int papi_err;

//set afiniti for CPU 1
cpu_set_t  mask;
CPU_ZERO(&mask);
CPU_SET(1, &mask);
int result = sched_setaffinity(0, sizeof(mask), &mask);  //0 for actual pid
if (result != 0){
	printf("affinity error");
	exit(-1);
}


if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
	fprintf(stderr, "PAPI is unsupported.\n");
	exit(-1);
}

if ((papi_err = PAPI_create_eventset(&eventSet)) != PAPI_OK) {
	fprintf(stderr, "Could not create event set: %s\n", PAPI_strerror(papi_err));
	PAPI_shutdown();
	exit(-1);
}


// initialize papi event set

for (i=0; i < 2; ++i) {
	if ((papi_err = PAPI_add_event(eventSet, events[i])) != PAPI_OK ) {
		fprintf(stderr, "Could not add event: %s\n", PAPI_strerror(papi_err));
		fflush(stderr);
	}
}

double first[SIZE][SIZE];
double second[SIZE][SIZE];
double multiply[SIZE][SIZE];
int iret;

for (j = 0; j < 4; j++){
    
    double *A;
    int i, j, n, ret;
    
    n = 3;
    A = calloc(n*n, sizeof(double));
    assert(A != NULL);
    
    A[IDX(0, 0, n)] = 4.0;   A[IDX(0, 1, n)] = 12.0;  A[IDX(0, 2, n)] = -16.0;
    A[IDX(1, 0, n)] = 12.0;  A[IDX(1, 1, n)] = 37.0;  A[IDX(1, 2, n)] = -43.0;
    A[IDX(2, 0, n)] = -16.0; A[IDX(2, 1, n)] = -43.0; A[IDX(2, 2, n)] = 98.0;

 
if ((papi_err = PAPI_start(eventSet)) != PAPI_OK) {
	fprintf(stderr, "Could not start counters: %s\n", PAPI_strerror(papi_err));
	PAPI_shutdown();
	exit(-1);
}
 /* Setup PAPI library and begin collecting data from the counters */
//if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
	//test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
	
	switch(j)
	{
	case 0:
        if (chol(A, 3)) {
            fprintf(stderr, "Error: matrix is either not symmetric or not positive definite.\n");
        } else {
            fprintf(stdout, "Tri(L) = \n");
            for (i = 0; i < 3; i++) {
                for (j = 0; j <= i; j++)
                    printf("%2.8lf\t", A[IDX(i, j, n)]);
                printf("\n");
            }
        }
		break;
        case 1:
            if (chol1(A, 3)) {
                fprintf(stderr, "Error: matrix is either not symmetric or not positive definite.\n");
            } else {
                fprintf(stdout, "Tri(L) = \n");
                for (i = 0; i < 3; i++) {
                    for (j = 0; j <= i; j++)
                        printf("%2.8lf\t", A[IDX(i, j, n)]);
                    printf("\n");
                }
            }
            break;
        case 2:
            if (chol2(A, 3)) {
                fprintf(stderr, "Error: matrix is either not symmetric or not positive definite.\n");
            } else {
                fprintf(stdout, "Tri(L) = \n");
                for (i = 0; i < 3; i++) {
                    for (j = 0; j <= i; j++)
                        printf("%2.8lf\t", A[IDX(i, j, n)]);
                    printf("\n");
                }
            }
            break;
        case 3:
            if (chol3(A, 3)) {
                fprintf(stderr, "Error: matrix is either not symmetric or not positive definite.\n");
            } else {
                fprintf(stdout, "Tri(L) = \n");
                for (i = 0; i < 3; i++) {
                    for (j = 0; j <= i; j++)
                        printf("%2.8lf\t", A[IDX(i, j, n)]);
                    printf("\n");
                }
            }
	default:
		return;


	}
	if ((papi_err = PAPI_stop(eventSet, values)) != PAPI_OK) {
		fprintf(stderr, "Could not get values: %s\n", PAPI_strerror(papi_err));
	}
  //if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
   // test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

 // printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
 // real_time, proc_time, flpins, mflops);
 // printf("%s\tPASSED\n", __FILE__);
 
    free(A);
    
	printf("Performance counters for factorization stage: \n");
	printf("\tFP OPS: %ld\n", values[0]);
	printf("\tLD INS: %ld\n", values[1]);
	
	fp_ops[j] = values[0];
	ld_ins[j] = values[1];

}

FILE *f = fopen("results.txt", "w");
if (f == NULL){
	printf("Error opening file!\n");	
	PAPI_shutdown();
	exit(1);
}

fprintf(f, "FP OPS\n\n");

for(j = 0; j < 9; j++){
	fprintf(f, "%lld\n", fp_ops[j]);
}

fprintf(f, "\n\nLD INS\n\n");

for(j = 0; j < 9; j++){
	fprintf(f, "%lld\n", ld_ins[j]);
}

fclose(f);
/* Initialize the Matrix arrays */
/*  for ( i=0; i<INDEX*INDEX; i++ ){
    mresult[0][i] = 0.0;
    matrixa[0][i] = matrixb[0][i] = rand()*(float)1.1; }
*/



  /* Matrix-Matrix multiply */
 /* for (i=0;i<INDEX;i++)
   for(j=0;j<INDEX;j++)
    for(k=0;k<INDEX;k++)
      mresult[i][j]=mresult[i][j] + matrixa[i][k]*matrixb[k][j];
*/
/* Collect the data into the variables passed in */
if ((papi_err = PAPI_start(eventSet)) != PAPI_OK) {
	fprintf(stderr, "Could not start counters: %s\n", PAPI_strerror(papi_err));
	PAPI_shutdown();
	exit(-1);
}

 PAPI_shutdown();
  exit(0);
}

static void test_fail(char *file, int line, char *call, int retval){
    printf("%s\tFAILED\nLine # %d\n", file, line);
    if ( retval == PAPI_ESYS ) {
        char buf[128];
        memset( buf, '\0', sizeof(buf) );
        sprintf(buf, "System error in %s:", call );
        perror(buf);
    }
    else if ( retval > 0 ) {
        printf("Error calculating: %s\n", call );
    }
    else {
        printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
    }
    printf("\n");
    exit(1);
}
