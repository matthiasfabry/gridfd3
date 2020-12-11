/*****************************************************************************/
#define THIS_IS "fd3 v.3.2 (Fabry, 8 September 2020)"
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_multimin.h>

#include "../mxfuns.h"
#include "fd3sep.h"
#include "../triorb.h"

/*****************************************************************************/

/* function and macro to kill the program with a short message */
#define FDBErrorString "\nError in fd3"
#define DIE(s) {fprintf(stderr,"%s: %s\n",FDBErrorString,s);fdbfailure();};
void fdbfailure(void) { exit ( EXIT_FAILURE ); }

/*****************************************************************************/

/* macros for reading values from keyboard */
#define GETDBL(x) {if(1!=scanf("%lg",x)) DIE("failed reading double");}
#define GETINT(x) {if(1!=scanf("%d", x)) DIE("failed reading int");}
#define GETLNG(x) {if(1!=scanf("%ld",x)) DIE("failed reading long");}
#define GETSTR(x) {if(1!=scanf("%s", x)) DIE("failed reading string");}

/*****************************************************************************/

#define SPEEDOFLIGHT 299792.458 /* speed of light in km/s */
#define TRIORB_NP 13

static char *triorb_strings[] = {
        /*  0 */  "period of AB--C",                    "day", "%.9lg",
        /*  1 */  "time of periast pass in AB--C",      "day", "%.9lg",
        /*  2 */  "eccentricity of AB--C",                "1", "%.9lg",
        /*  3 */  "periast long of AB in AB--C",        "deg", "%.9lg",
        /*  4 */  "a sin(incl) of AB in AB--C",   "light-day", "%.9lg",
        /*  5 */  "a sin(incl) of C in AB--C",    "light-day", "%.9lg",
        /*  6 */  "period A--B",                        "day", "%.9lg",
        /*  7 */  "time of periast pass in A--B",       "day", "%.9lg",
        /*  8 */  "eccentricity of A--B",                 "1", "%.9lg",
        /*  9 */  "periast long of A in A--B",          "deg", "%.9lg",
        /* 10 */  "rv semiamp of A in A--B",           "km/s", "%.9lg",
        /* 11 */  "rv semiamp of B in A--B",           "km/s", "%.9lg",
        /* 12 */  "periast long adv per cycle in A--B", "deg", "%.9lg"
};

static char *triorb_pname   (long j) { return triorb_strings[3*j];   }
static char *triorb_punit   (long j) { return triorb_strings[3*j+1]; }
static char *triorb_pformat (long j) { return triorb_strings[3*j+2]; }

/*****************************************************************************/

static long   K, M, N, Ndft, ksw[3], nfp, opsw[TRIORB_NP];
static double **dftobs, **dftmod, **dftres;
static double rvstep, *otimes, *rvcorr, *sig, **lfm, **rvm;
static double op0[TRIORB_NP], dop0[TRIORB_NP];
static double meritfngsl ( const gsl_vector *v, void *params );
static double meritfn ( double *op );

#define MX_FDBINARY_FORMAT "%15.8E   "
static char *mxfd3fmts=MX_FDBINARY_FORMAT;

/* fd3 */

int main ( void ) {

    long i, i0, i1, j, k, vc, vlen, nruns, niter, rootfnlen;
    double **masterobs, **mod, **res, **obs, z0, z1, stoprat;
    char rootfn[1024], obsfn[1024], resfn[1024];
    char modfn[1024], rvsfn[1024], logfn[1024];
    char *starcode[] = {"A","B","C"};
    FILE *logfp;

    setbuf ( stdout, NULL );
    MxError( FDBErrorString, stdout, fdbfailure );
    MxFormat( mxfd3fmts );

    printf ( "\n  %s\n", THIS_IS );

    printf ( "\n  SECTION 1: LOADING OBSERVED SPECTRA\n\n" );
    printf ( "  observed spectra master file = " );
    GETSTR ( obsfn ); printf ( "%s\n", obsfn );
    vc=0; vlen=0; masterobs = MxLoad ( obsfn, &vc, &vlen ); M = vc - 1;
    printf ( "  loaded %ld spectra with %ld bins per spectrum\n", M, vlen );
    z0 = **masterobs; z1 = *(*masterobs+vlen-1);
    rvstep = SPEEDOFLIGHT * ( - 1 + exp ((z1-z0)/(vlen-1)) );
    printf ( "  lnlambda range from %.9g to %.9g\n", z0, z1 );
    printf ( "  rv step %lg km/s per bin\n", rvstep );
    printf ( "  use data from lnlambda = " );
    GETDBL ( &z0 ); printf ( "%.9g\n", z0 );
    printf ( "             to lnlambda = " );
    GETDBL ( &z1 ); printf ( "%.9g\n", z1 );
    i0 = 0; while ( *(*masterobs+i0) < z0 ) i0++;
    i1 = vlen-1; while ( z1 < *(*masterobs+i1) ) i1--;
    N = i1 - i0 + 1;
    printf ( "  selected %ld bins per spectrum\n", N );
    obs = MxAlloc ( M+1, N );
    for ( i = 0 ; i < N ; i++ ) for ( j = 0 ; j <= M ; j++ )
            *(*(obs+j)+i) = *(*(masterobs+j)+i0+i);
    MxFree ( masterobs, vc, vlen );

    printf ( "\n  SECTION 2: OUTPUT FILES\n\n" );

    /* rio news */
    printf ( "  output files root name = " );
    GETSTR ( rootfn ); printf ( "%s\n\n", rootfn );
    rootfnlen = strlen ( rootfn );

    sprintf ( obsfn, "%s", rootfn ); sprintf ( obsfn+rootfnlen, "%s", ".obs" );
    printf ( "  used part of observed spectra ------------> \"%s\"\n", obsfn );
    MxWrite ( obs, M+1, N, obsfn );

    sprintf ( modfn, "%s", rootfn ); sprintf ( modfn+rootfnlen, "%s", ".mod" );
    printf ( "  model spectra ----------------------------> \"%s\"\n", modfn );

    sprintf ( resfn, "%s", rootfn ); sprintf ( resfn+rootfnlen, "%s", ".res" );
    printf ( "  residuals (obs. - calc.) -----------------> \"%s\"\n", resfn );

    sprintf ( rvsfn, "%s", rootfn ); sprintf ( rvsfn+rootfnlen, "%s", ".rvs" );
    printf ( "  radial velocities (in bin units) ---------> \"%s\"\n", rvsfn );

    sprintf ( logfn, "%s", rootfn ); sprintf ( logfn+rootfnlen, "%s", ".log" );
    if ( NULL == ( logfp = fopen ( logfn, "w" ) ) )
    DIE ( "could not open log file" );
    setbuf ( logfp, NULL );
    printf ( "  optimisation log -------------------------> \"%s\"\n", logfn );

    printf ( "\n  SECTION 3: COMPONENT SPECTRA MODELING SWITCHES (0/1)\n\n" );
    for ( K = i = 0 ; i < 3 ; i++ ) {
        int sw;
        GETINT(&sw);
        sw = sw ? 1 : 0;
        printf ( "  %s = %d\n", starcode[i], sw );
        if ( sw ) ksw[K++] = i;
    }
    printf ( "\n  number of components to be resolved is %ld\n", K );

    /* allocating memory */
    Ndft = 2*(N/2 + 1);
    dftobs = MxAlloc ( M, Ndft ); dft_fwd ( M, N, obs+1, dftobs );
    otimes = *MxAlloc ( 1, M );
    rvcorr = *MxAlloc ( 1, M );
    sig = *MxAlloc ( 1, M );
    res = MxAlloc ( M+1, N ); dftres = MxAlloc ( M, Ndft );
    mod = MxAlloc ( K+1, N ); dftmod = MxAlloc ( K, Ndft );
    rvm = MxAlloc ( K, M );
    lfm = MxAlloc ( K, M );
    for ( i = 0 ; i < N ; i++ ) { *(*res+i) = *(*obs+i); *(*mod+i) = *(*obs+i); }

    printf ( "\n  SECTION 4: DESCRIPTORS TO OBSERVED SPECTRA\n" );
    printf ( "             AND LIGHT-FACTORS ASSIGNED TO COMPONENTS\n\n" );
    printf ( "  %14s%12s%12s", "t_obs", "rv_corr", "noise_rms" );
    for ( k = 0 ; k < K ; k++ ) printf ( "      lf_%s", starcode[ksw[k]] );
    printf ( "\n" );
    printf ( "  %14s%12s%12s", "[day]", "[km/s]", "[1]" );
    for ( k = 0 ; k < K ; k++ ) printf ( "       [1]" );
    printf ( "\n\n" );
    for ( j = 0 ; j < M ; j++ )
    {
        GETDBL(otimes+j); printf ( "  %14.5lf", *(otimes+j) );
        GETDBL(rvcorr+j); printf ( "%12.4lf", *(rvcorr+j) );
        GETDBL(sig+j); printf ( "%12.4lf", *(sig+j) );
        for ( k = 0; k < K ; k++ )
        { GETDBL(*(lfm+k)+j); printf ( "%10.4lf", *(*(lfm+k)+j) ); }
        putchar ( '\n' );
    }

    printf ( "\n  SECTION 5: PARAMETERS OF ORBITS\n" );
    for ( nfp = i = 0 ; i < TRIORB_NP ; i++ ) {
        if ( 0 == i ) printf ( "\n  wide (AB--C) orbit\n\n" );
        if ( 6 == i ) printf ( "\n  tight (A--B) orbit\n\n" );
        printf ( "  %s [%s] = ", triorb_pname(i), triorb_punit(i) );
        GETDBL(op0+i); printf ( triorb_pformat(i), *(op0+i) );
        printf ( " +/- " );
        GETDBL(dop0+i); printf ( triorb_pformat(i), *(dop0+i) );
        if ( *(dop0+i) ) { opsw[nfp++] = i; printf ( " *** free ***" ); }
        printf ( "\n" );
    }
    printf ( "\n  number of free orbital parameters is %ld\n", nfp );

    printf ( "\n  SECTION 6: OPTIMISATION DETAILS\n\n" );
    printf ( "no optimisation possible in this build, only separation\n");

    printf ( "\n  SECTION 7: Separation\n\n" );

    double chi2;

    printf ( "  WARNING: *** no free orbital parameters *** \n" );
    chi2 = meritfn ( op0 );
    printf ( "  separation at the starting point:  chi2=%lg  gof=%.2lf\n",
             chi2, gsl_sf_gamma_inc_Q ( N*(M-K)/2.0, chi2/2.0 ) );
    dft_bck ( K, N, dftmod, mod+1 ); MxWrite ( mod, K+1, N, modfn );
    dft_bck ( M, N, dftres, res+1 ); MxWrite ( res, M+1, N, resfn );
    MxWrite( rvm, K, M, rvsfn );

    printf ( "  EXITING REGULARLY\n\n" );

    return EXIT_SUCCESS;

}

/*****************************************************************************/

double meritfn ( double *opin )
{

    long j, k;
    double op[TRIORB_NP], rv[3];

    op[ 0] = opin[ 0];
    op[ 1] = opin[ 1];
    op[ 2] = opin[ 2];
    op[ 3] = opin[ 3] * (M_PI/180);
    op[ 4] = opin[ 4];
    op[ 5] = opin[ 5];
    op[ 6] = opin[ 6];
    op[ 7] = opin[ 7];
    op[ 8] = opin[ 8];
    op[ 9] = opin[ 9] * (M_PI/180);
    op[10] = opin[10] / rvstep;
    op[11] = opin[11] / rvstep;
    op[12] = opin[12] * (M_PI/180);

    for ( j = 0 ; j < M ; j++ ) {
        triorb_rv ( op, otimes[j], rv );
        for ( k = 0 ; k < K ; k++ )
            *(*(rvm+k)+j) = rv[ksw[k]] + *(rvcorr+j) / rvstep;
    }

    return fd3sep ( K, M, N, dftobs, sig, rvm, lfm, dftmod, dftres );
}

/*****************************************************************************/

