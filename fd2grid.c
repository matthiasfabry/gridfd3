/*****************************************************************************/
#define THIS_IS "fd2grid v.1.1 (Fabry, 16 Dec 2019)"
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_multimin.h>

#include "mxfuns.h"
#include "fd2sep.h"
#include "orb.h"

/*****************************************************************************/

/* function and macro to kill the program with a short message */
#define FDBErrorString "\nError in fd2grid"
#define DIE(s) {fprintf(stderr,"%s: %s\n",FDBErrorString,s);fdbfailure();};
void fdbfailure(void) { exit ( EXIT_FAILURE ); }

/*****************************************************************************/

/* macros for reading values from keyboard */
#define GETDBL(x) {if(1!=scanf("%lg",x)) DIE("failed reading double");}
#define GETINT(x) {if(1!=scanf("%d", x)) DIE("failed reading int");}
#define GETLNG(x) {if(1!=scanf("%ld",x)) DIE("failed reading long");}
#define GETSTR(x) {if(1!=scanf("%s", x)) DIE("failed reading string");}

/*****************************************************************************/

#define SPEEDOFLIGHT 2.998E5 /* speed of light in km/s */

/*****************************************************************************/

static long   K, M, N, Ndft, nfp;
static double **dftobs;
static double rvstep, *otimes, *rvcorr, *sig, **lfm, **rvm;
static double op0[ORB_NP];
static double meritfn ( double *op , double rvA, double rvB);

#define MX_FDBINARY_FORMAT "%15.8E   "
static char *mxfd3fmts=MX_FDBINARY_FORMAT;

/* fd3 */

int main ( int argc, char *argv[] ) {

	long i, i0, i1, j, k, vc, vlen;
	double **masterobs, **obs, z0, z1;
	char obsfn[1024];

	setbuf ( stdout, NULL );
	MxError( FDBErrorString, stdout, fdbfailure );
	MxFormat( mxfd3fmts );
	GETSTR ( obsfn );
	vc=0;
	vlen=0;
	masterobs = MxLoad ( obsfn, &vc, &vlen );
	M = vc - 1;
	z0 = **masterobs;
	z1 = *(*masterobs+vlen-1);
	rvstep = SPEEDOFLIGHT * ( - 1 + exp ((z1-z0)/(vlen-1)) );
	GETDBL ( &z0 );
	GETDBL ( &z1 );
	i0 = 0;
	while ( *(*masterobs+i0) < z0 )
		i0++;
	i1 = vlen-1;
	while ( z1 < *(*masterobs+i1) )
		i1--;
	N = i1 - i0 + 1;
	obs = MxAlloc ( M+1, N );
	for ( i = 0 ; i < N ; i++ )
		for ( j = 0 ; j <= M ; j++ )
			*(*(obs+j)+i) = *(*(masterobs+j)+i0+i);
	MxFree ( masterobs, vc, vlen );
	K = 2;

	/* allocating memory */
	Ndft = 2*(N/2 + 1);
	dftobs = MxAlloc ( M, Ndft );
	dft_fwd ( M, N, obs+1, dftobs );
	otimes = *MxAlloc ( 1, M );
	rvcorr = *MxAlloc ( 1, M );
	sig = *MxAlloc ( 1, M );
	rvm = MxAlloc ( K, M );
	lfm = MxAlloc ( K, M );
	for ( j = 0 ; j < M ; j++ ) {
		GETDBL(otimes+j);
		GETDBL(rvcorr+j);
		GETDBL(sig+j);
		for ( k = 0; k < K ; k++ ) {
			GETDBL(*(lfm+k)+j);
		}
	}

	for ( nfp = i = 0 ; i < ORB_NP ; i++ ) {
		GETDBL(op0+i);
	}

	double lowA, highA, lowB, highB;
	double stepA, stepB;

	GETDBL(&lowA);
	GETDBL(&highA);
	GETDBL(&stepA);
	GETDBL(&lowB);
	GETDBL(&highB);
	GETDBL(&stepB);

	int sampA, sampB;
	double *rvAs, *rvBs;
	double **chi2;

	sampA = (highA - lowA) / stepA + 1;
	sampB = (highB - lowB) / stepB + 1;

	rvAs = *MxAlloc(1, sampA);
	for (i=0; i<sampA; i++){
		*(rvAs+i) = lowA + i*stepA;
	}
	rvBs = *MxAlloc(1, sampB);
	for (i=0;i<sampB; i++){
		*(rvBs+i) = lowB + i*stepB;
	}
	chi2 = MxAlloc(sampA, sampB);

	// here is where the heavy lifting occurs
	printf ( "k1 k2 chisq \n" );
	for (i=0; i<sampA; i++){
		for (j=0; j<sampB; j++){
			*(*(chi2+i)+j) = meritfn ( op0 , *(rvAs+i), *(rvBs+j));
			printf ( "%.5f %.5f %.5f\n", *(rvAs+i), *(rvBs+j), *(*(chi2+i)+j));
		}
	}

	return EXIT_SUCCESS;
}

/*****************************************************************************/

double meritfn ( double *opin, double rvA, double rvB ) {

	long j, k;
	double op[ORB_NP+2], rv[2];

	op[0] = opin[0];
	op[1] = opin[1];
	op[2] = opin[2];
	op[3] = opin[3] * (M_PI/180);
	op[4] = rvA / rvstep;
	op[5] = rvB / rvstep;

	for ( j = 0 ; j < M ; j++ ) {
		orb_rv ( op, otimes[j], rv );
		for ( k = 0 ; k < K ; k++ )
			*(*(rvm+k)+j) = rv[k] + *(rvcorr+j) / rvstep;
	}

	return fd2sep ( K, M, N, dftobs, sig, rvm, lfm);
}

/*****************************************************************************/
