#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <float.h>
#include "base_types.h"
#include "FloatModel.h"
#include "FeaturePool.h"
#include "IvDecoder.h"

// #define _HARDCODE_TV
// #define _TV_DBG_MSG

extern int mat_inv(const double *x, int n, double *a);

// eps(x) for single input
static double eps(double x)
{
	double r;
	int exponent;

	if (x < 0.0) 
		x = -x;
	if (x != x)	// test if x is Not-a-Number
		return x;
	
	if (x <= 1.17549435E-38F) 
		r = 1.4013E-45F;
	else {
		r = frexp(x, &exponent);
		r = ldexp(1.0, exponent - 24);
    }
	return r;
}

// *   m = row dimension of a
// *   n = column dimension of a
// Output: X
static void pinv(double *a, int m, int n, double *X)
{
	int i, j, k, r1, pos = 0;
	double t, norm;
	double *w, *v_mem;
	double **v, **ppa, **pX;
	double *mem, **ppmem;

	mem = (double *)malloc(sizeof(double) * (m + m * n));
	w     = mem;
	v_mem = mem + m;

	ppmem = (double **)malloc(sizeof(double *) * m * 3);
	v   = ppmem + pos, pos += m;
	ppa = ppmem + pos, pos += m;
	pX  = ppmem + pos, pos += m;

	for (i = 0; i < m; ++i) {
		v[i]   = &v_mem[i * n];
		ppa[i] = &a[i * n];
		pX[i]  = &X[i * n];
	}

	dsvd(ppa, m, n, w, v);

	// norm(w, inf)
	norm = (w[0] < 0) ? (-w[0]) : (w[0]);
	for (i = 1; i < m; ++i) {
		t = (w[i] < 0) ? (-w[i]) : (w[i]); 
		if (t > norm) norm = t;
	}
		
	// t = tolerance, any singular values less than a tolerance are treated as zero
	t = max(m, n) * eps(norm);
	for (i = 0, r1 = 0; i < m; ++i) {
		if (t > w[i])
			++r1;
	}	
	// 從r1起的columns都去掉
	
	// w = 1./w
	for (i = 0; i < (m - r1); ++i)
		w[i] = 1 / w[i];
	 
	for (i = 0; i < (m - r1); ++i) {
		for (j = 0; j < n; ++j)
			v[j][i] *= w[i];	// bsxfun(@times,V,s.')
	}

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			double sum = 0;
			for (k = 0; k < (m - r1); ++k)
				sum += v[i][k] * ppa[j][k];
			pX[i][j] = sum;
		}
	}

	free(mem);
	free(ppmem);
}

double randn(double mu, double sigma)
{
	const double epsilon = -1e10;
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0 = 0, z1 = 0;
	static bool generate = false;
	double u1, u2;

	generate = !generate;
	if (!generate)
		return z1 * sigma + mu;

	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	}
	while ( u1 <= epsilon );

	if (u1 == 0) {
		z0 = 0;
		z1 = 0;
	}
	else {
		z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
		z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	}

	return z0 * sigma + mu;
}

// compute the posterior means and covariance matrices of the factors or latent variables
// output: T
static int expectation_tv(double **T, double **N, double **F, float *S, int nFile, int tv_dim, int nGauss, int nDim)
{
	int i, j, k, l, n, idx, size, pos = 0;
	double sum;
	double *mem;
	double *T_invS_mem, *LU, *RU, *Ex, *Exx, *L, *Cxx, *B;
	double *tmp, *tmp1, *tmp2, *pLU, *pLU_inv;
	double **T_invS;

	n = nGauss * nDim;
// 	size = (tv_dim * n) + (tv_dim * tv_dim * nGauss) + (tv_dim * n) + (tv_dim * nFile) + (tv_dim * tv_dim * nFile)
// 		   + (tv_dim * tv_dim) + (tv_dim * tv_dim) + tv_dim + (tv_dim * n) + (tv_dim * tv_dim * nFile)
// 		   + (tv_dim * tv_dim);
	
	size = tv_dim * (n + n + nFile + 1 + n) + (tv_dim * tv_dim * (nGauss + nFile + nFile + 3));
	
	mem = (double *)calloc(size, sizeof(double));
	T_invS_mem = mem + pos, pos += tv_dim * n;
	LU         = mem + pos, pos += tv_dim * tv_dim * nGauss;
	RU         = mem + pos, pos += tv_dim * n;
	Ex         = mem + pos, pos += tv_dim * nFile;
	Exx        = mem + pos, pos += tv_dim * tv_dim * nFile;
	L          = mem + pos, pos += tv_dim * tv_dim;
	Cxx        = mem + pos, pos += tv_dim * tv_dim;
	B          = mem + pos, pos += tv_dim;
	tmp        = mem + pos, pos += tv_dim * n;

	T_invS = (double **)malloc(sizeof(double *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		T_invS[i] = &T_invS_mem[i * n];

	for (i = 0; i < n; ++i) {
		for (j = 0; j < tv_dim; ++j)
			T_invS[j][i] = T[j][i] / S[i];		// T_invS = bsxfun(@rdivide, T, S');
	}

	for (i = 0; i < nFile; ++i) {
		for (j = 0; j < tv_dim; ++j) {
			for (k = 0; k < n; ++k) {
				tmp[j * n + k] = T_invS[j][k] * N[i][k / nDim];  // tmp = bsxfun(@times, T_invS, N1(ix, idx_sv));
			}
		}
		for (j = 0; j < tv_dim; ++j) {
			for (k = 0; k < tv_dim; ++k) {
				sum = 0;
				for (l = 0; l < n; ++l) {
					sum += tmp[k * n + l] * T[j][l];  // tmp * T'
				}

				if (j == k) sum += 1;			// 對角線 +1
				L[j * tv_dim + k] = sum;
			}
		}

#ifdef _TV_DBG_MSG
		printf("pinv input: \n");
		for (j = 0; j < tv_dim; ++j)
			printf("%g ", L[j]);
		printf("\n");
#endif

		pinv(L, tv_dim, tv_dim, Cxx);			// Cxx = pinv(L)	

#ifdef _TV_DBG_MSG
		printf("pinv output: \n");
		for (j = 0; j < tv_dim; ++j)
			printf("%g ", Cxx[j]);
		printf("\n");
#endif

		for (j = 0; j < tv_dim; ++j) {
			sum = 0;
			for (k = 0; k < n; ++k) {
				sum += T_invS[j][k] * F[i][k];	//  B = T_invS * F1(ix, :)';
			}
			B[j] = sum;
		}

		for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
			for (k = 0; k < tv_dim; ++k) {
				sum += Cxx[j * tv_dim + k] * B[k];
			}
			Ex[j * nFile + i] = sum;			//  Ex(:, ix) = Cxx * B;  this is the posterior mean E[x]
		}
		
		idx = i * tv_dim * tv_dim;
		for (j = 0; j < tv_dim; ++j) {
			for (k = 0; k < tv_dim; ++k) {	// Exx(:, :, ix) = Cxx + Ex(:, ix) * Ex(:, ix)';
 				Exx[idx + j * tv_dim + k] = Cxx[j * tv_dim + k] + Ex[j * nFile + i] * Ex[i * nFile + k];
			}
		}
	}

	// RU = RU + Ex * F1;
	for (i = 0; i < tv_dim; ++i) {
		for (j = 0, sum = 0; j < n; ++j, sum = 0) {
			for (k = 0; k < nFile; ++k)
				sum += Ex[i * nFile + k] * F[k][j];
			RU[i * n + j] += sum;
		}
	}

	tmp1 = mem + pos, pos += tv_dim * tv_dim * nFile;
	n = tv_dim * tv_dim;
	for (i = 0; i < nGauss; ++i) {
		for (j = 0; j < nFile; j++) {
			idx = j * n;
			for (k = 0; k < n; ++k) {
				tmp1[idx + k] = Exx[idx + k] * N[j][i];	// tmp = bsxfun(@times, Exx, reshape(N1(:, mix),[1 1 len]));
			}
		}

		idx = i * n;
		for (j = 0, sum = 0; j < n; ++j, sum = 0) {
			for (k = 0; k < nFile; k++)
				sum += tmp1[k * n + j];
			LU[idx + j] += sum;					// LU{mix} = LU{mix} + sum(tmp, 3);
		}
#ifdef _TV_DBG_MSG
		printf("LU{%d} = %g\n", i+1, LU[idx]);
#endif
	}
	
	/*** maximization_tv ***/
	idx = 0;
	n = nGauss * nDim;	// n == RU num rows.
	tmp2 = T_invS_mem;
	pLU = LU;
	pLU_inv = mem + pos, pos += tv_dim * tv_dim;

	for (i = 0; i < nGauss; ++i, pLU += (tv_dim * tv_dim), 	idx += nDim) 
	{
		for (j = 0; j < tv_dim; ++j)
			memcpy(&tmp[j * nDim], &RU[j * n + idx], sizeof(double) * nDim);

		memcpy(pLU_inv, pLU, sizeof(double) * tv_dim * tv_dim);
		mat_inv(pLU_inv, tv_dim, pLU_inv);

		for (j = 0; j < tv_dim; ++j) {
			for (k = 0, sum = 0; k < nDim; ++k, sum = 0) {
				for (l = 0; l < tv_dim; ++l)
					sum += pLU_inv[j * tv_dim + l] * tmp[l * nDim + k];
				tmp2[j * nDim + k] = sum;
			}
		}

		// tmp2 copy to RU
		for (j = 0; j < tv_dim; ++j)
			memcpy(&RU[j * n + idx], &tmp2[j * nDim], sizeof(double) * nDim);
	}

	// Done. result store in T
	for (i = 0; i < tv_dim; ++i)
		memcpy(T[i], &RU[i * n], sizeof(double) * n);

	SAFE_FREE(T_invS);
	SAFE_FREE(mem);
	return 0;
}

#ifdef _HARDCODE_TV
void LoadT(double **T, int m, int n)
{
	int i, j;
	char ctmp[64];
	FILE *fp = fopen("T.txt", "r");

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			fscanf(fp, "%s", ctmp);
			T[i][j] = atof(ctmp);
		}
	}
	fclose(fp);
}
#endif

void TrainTVSpace(GMMDATA *pUbmData, double **NF, int nFile, int tv_dim, int nIter, double *T)
{
	int i, n;
	int nDim = pUbmData->m_nDim;
	int nGauss = pUbmData->m_nNumGMMGauss;
	double **N = NULL, **F = NULL, **ppT = NULL;
	double sum_sigma = 0;
	float *S = &pUbmData->m_pfMemPool[nDim * nGauss];
	float m = 1e-30f;
	
	n = nDim * nGauss;
	for(i = 0; i < n; ++i) {
		if (m < S[i]) m = S[i];
		sum_sigma += S[i];
	}
// 	sum_sigma /= m;
	// 2017.5.5 note: 這個scale會影響到Train TV穩定度
	while (sum_sigma > 0.01) sum_sigma /= 10;	

	N = (double **)malloc(sizeof(double *) * nFile);
	F = (double **)malloc(sizeof(double *) * nFile);

	for (i = 0; i < nFile; ++i) {
		N[i] = &NF[i][0];
		F[i] = &NF[i][nGauss];
	}

	n = tv_dim * nDim * nGauss;
	ppT = (double **)malloc(sizeof(double *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		ppT[i] = &T[i * nDim * nGauss];
	
#ifdef _HARDCODE_TV
	LoadT(ppT, tv_dim, nDim * nGauss);
 	for (i = 0; i < n; ++i) 
 		T[i] *= sum_sigma;
#else
	srand((unsigned)time(NULL));
	for (i = 0; i < n; ++i)
		T[i] = randn(0, 1) * sum_sigma;

// 	n /= tv_dim;
// 	int nt = n / 2;
// 	for (i = 0; i < n; ++i) {
// 		for (int j = 0; j < tv_dim; ++j)
// 			T[i + (j * n)] = ((double)(-nt+i+1) / n) * sum_sigma;
// 	}		
#endif
	
	for (i = 0; i < nIter; ++i)
		expectation_tv(ppT, N, F, S, nFile, tv_dim, nGauss, nDim);

	SAFE_FREE(N);
	SAFE_FREE(F);
	SAFE_FREE(ppT);
}

void ExtractIVector(GMMDATA *pUbmData, double *NF, int tv_dim, double *T, double *iv)
{
	int i, j, k, l, n, size, pos = 0;
	int nDim = pUbmData->m_nDim;
	int nGauss = pUbmData->m_nNumGMMGauss;
	double sum = 0;
	double *mempool = NULL;
	double *N = NULL, *F = NULL, **ppT = NULL;
	double *L = NULL, *Cxx = NULL, *B = NULL, *tmp = NULL;
	double *T_invS_mem = NULL, **T_invS = NULL;
	float *S = &pUbmData->m_pfMemPool[nDim * nGauss];
	
	n = nDim * nGauss;

	N = NF;
	F = NF + nGauss;
	ppT = (double **)malloc(sizeof(double *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		ppT[i] = &T[i * n];	// T[tv_dim][nDim * nGauss]

	// size = T_invS_mem + L + Cxx + B + tmp
	size = (tv_dim * n) + (tv_dim * tv_dim) + (tv_dim * tv_dim) + tv_dim + (tv_dim * n);
	mempool = (double *)malloc(sizeof(double) * size);

	T_invS_mem = mempool, pos += tv_dim * n;
	L   = mempool + pos, pos += tv_dim * tv_dim;
	Cxx = mempool + pos, pos += tv_dim * tv_dim;
	B   = mempool + pos, pos += tv_dim;
	tmp = mempool + pos, pos += tv_dim * n;

	T_invS = (double **)malloc(sizeof(double *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		T_invS[i] = &T_invS_mem[i * n];

	for (i = 0; i < n; ++i) {
		for (j = 0; j < tv_dim; ++j)
			T_invS[j][i] = ppT[j][i] / S[i];	// T_invS = bsxfun(@rdivide, T, S');
	}

	for (j = 0; j < tv_dim; ++j) {
		for (k = 0; k < n; ++k) {
			tmp[j * n + k] = T_invS[j][k] * N[k / nDim];  // bsxfun(@times, T_invS, N(idx_sv)')
		}
	}
	for (j = 0; j < tv_dim; ++j) {
		for (k = 0, sum = 0; k < tv_dim; ++k, sum = 0) {
			for (l = 0; l < n; ++l)
				sum += tmp[k * n + l] * ppT[j][l];  // tmp * T'

			if (j == k) sum += 1;			// 對角線 +1
			L[j * tv_dim + k] = sum;
		}
	}
	pinv(L, tv_dim, tv_dim, Cxx);			// Cxx = pinv(L)	

	for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
		for (k = 0; k < n; ++k)
			sum += T_invS[j][k] * F[k];	//  B = T_invS * F;
		B[j] = sum;
	}

	for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
		for (k = 0; k < tv_dim; ++k)
			sum += Cxx[j * tv_dim + k] * B[k];
		iv[j] = sum;			//  x = pinv(L) * B;
	}

	SAFE_FREE(mempool);
	SAFE_FREE(ppT);	
	SAFE_FREE(T_invS);
}
