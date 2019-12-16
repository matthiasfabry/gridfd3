
#include <stdlib.h>
#include <math.h>

#include "kepler.h"
#include "orb.h"

#define LITESPEED 2.998E05

int orb_rv ( double *op, double t, double *rv ) {

	double tp, tt, te, toA, tkA, tkB;
	double trvA, rv_A, rv_B;

	tp   = op[0];
	tt   = op[1];
	te   = op[2];
	toA  = op[3];
	tkA  = op[4];
	tkB  = op[5];

	trvA = kepler_rv ( 2 * M_PI * ( t - tt ) / tp, te, toA );

	rv_A =   tkA * trvA;
	rv_B = - tkB * trvA;

	/* finish */

	*(rv+0) = rv_A;
	*(rv+1) = rv_B;

	return EXIT_SUCCESS;
}
