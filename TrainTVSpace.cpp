#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "base_types.h"
#include "FloatModel.h"
#include "FeaturePool.h"
#include "IvDecoder.h"

// #define _HARDCODE_TEST

#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308 /* max value */
#endif

extern void inv(float *X, int n);

// *   m = row dimension of a
// *   n = column dimension of a
// Output: X
static void pinv(float *a, int m, int n, float *X)
{
	int i, j, k, pos = 0;
	float t;
	float *w, *v_mem;
	float **v, **ppa, **pX;
	float *mem, **ppmem;

	mem = (float *)malloc(sizeof(float) * (m + m * n));
	w     = mem;
	v_mem = mem + m;

	ppmem = (float **)malloc(sizeof(float *) * m * 3);
	v   = ppmem + pos, pos += m;
	ppa = ppmem + pos, pos += m;
	pX  = ppmem + pos, pos += m;

	for (i = 0; i < m; ++i) {
		v[i]   = &v_mem[i * n];
		ppa[i] = &a[i * n];
		pX[i]  = &X[i * n];
	}

	dsvd(ppa, m, n, w, v);

	// swap for match the format in matlab
	for (i = 0, j = m-1; i < (m >> 1); ++i, --j) {
		t = w[i];
		w[i] = 1 / w[j];
		w[j] = 1 / t;		// s = 1./s
	}
	for (k = 0; k < n; ++k) {
		for (i = 0, j = m-1; i < (m >> 1); ++i, --j) {
			t = ppa[k][i];
			ppa[k][i] = ppa[k][j];
			ppa[k][j] = t;
			t = v[k][i];
			v[k][i] = v[k][j];
			v[k][j] = t;
		}
	}

// 	for (i = 0; i < m; ++i) {
// 		float wt = (w[i] < 0) ? (-w[i]) : (w[i]);
// 		if (wt > tol) tol = wt;		// find eps(norm(s,inf))
// 	}
// 	float tol = m * FLT_EPSILON;
	 
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j)
			v[j][i] *= w[i];	// bsxfun(@times,V,s.')
	}

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			float sum = 0;
			for (k = 0; k < n; ++k)
				sum += v[i][k] * ppa[j][k];
			pX[i][j] = sum;
		}
	}

	free(mem);
	free(ppmem);
}

// compute the posterior means and covariance matrices of the factors or latent variables
// output: T
static int expectation_tv(float **T, float **N, float **F, float *S, int nFile, int tv_dim, int nGauss, int nDim)
{
	int i, j, k, l, n, idx, size, pos = 0;
	double sum;
	float *mem;
	float *T_invS_mem, *LU, *RU, *Ex, *Exx, *L, *Cxx, *B;
	float *tmp, *tmp1, *tmp2, *pLU, *pLU_inv;
	float **T_invS;

	n = nGauss * nDim;
// 	size = (tv_dim * n) + (tv_dim * tv_dim * nGauss) + (tv_dim * n) + (tv_dim * nFile) + (tv_dim * tv_dim * nFile)
// 		   + (tv_dim * tv_dim) + (tv_dim * tv_dim) + tv_dim + (tv_dim * n) + (tv_dim * tv_dim * nFile)
// 		   + (tv_dim * tv_dim);
	
	size = tv_dim * (n + n + nFile + 1 + n) + (tv_dim * tv_dim * (nGauss + nFile + nFile + 3));
	
	mem = (float *)calloc(size, sizeof(float));
	T_invS_mem = mem + pos, pos += tv_dim * n;
	LU         = mem + pos, pos += tv_dim * tv_dim * nGauss;
	RU         = mem + pos, pos += tv_dim * n;
	Ex         = mem + pos, pos += tv_dim * nFile;
	Exx        = mem + pos, pos += tv_dim * tv_dim * nFile;
	L          = mem + pos, pos += tv_dim * tv_dim;
	Cxx        = mem + pos, pos += tv_dim * tv_dim;
	B          = mem + pos, pos += tv_dim;
	tmp        = mem + pos, pos += tv_dim * n;

	T_invS = (float **)malloc(sizeof(float *) * tv_dim);

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
					if (sum > FLT_MAX || sum < -FLT_MAX + 0.001) 
						sum = sum;
				}

				if (j == k) sum += 1;			// 對角線 +1
				L[j * tv_dim + k] = (float)sum;
			}
		}
		pinv(L, tv_dim, tv_dim, Cxx);			// Cxx = pinv(L)	

		for (j = 0; j < tv_dim; ++j) {
			sum = 0;
			for (k = 0; k < n; ++k) {
				sum += T_invS[j][k] * F[i][k];	//  B = T_invS * F1(ix, :)';
				if (sum > FLT_MAX || sum < -FLT_MAX + 0.001) 
					sum = sum;
			}
			B[j] = (float)sum;
		}

		for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
			for (k = 0; k < tv_dim; ++k) {
				sum += Cxx[j * tv_dim + k] * B[k];
				if (sum > FLT_MAX || sum < -FLT_MAX + 0.001) 
					sum = sum;
			}
			Ex[j * nFile + i] = (float)sum;			//  Ex(:, ix) = Cxx * B;  this is the posterior mean E[x]
		}
		
		idx = i * tv_dim * tv_dim;
		for (j = 0; j < tv_dim; ++j) {
			for (k = 0; k < tv_dim; ++k) {	// Exx(:, :, ix) = Cxx + Ex(:, ix) * Ex(:, ix)';
				sum = Cxx[j * tv_dim + k] + Ex[j * nFile + i] * Ex[i * nFile + k];
				if (DEBUG_CHECK_LIMIT(sum)) 
					sum = sum;
				CHECK_FLT_LIMIT(sum);
				Exx[idx + j * tv_dim + k] = (float)sum;
// 				Exx[idx + j * tv_dim + k] = Cxx[j * tv_dim + k] + Ex[j * nFile + i] * Ex[i * nFile + k];
			}
		}
	}

	// RU = RU + Ex * F1;
	for (i = 0; i < tv_dim; ++i) {
		for (j = 0, sum = 0; j < n; ++j, sum = 0) {
			for (k = 0; k < nFile; ++k)
				sum += Ex[i * nFile + k] * F[k][j];
			RU[i * n + j] += (float)sum;
			if(DEBUG_CHECK_LIMIT(sum))
				sum = sum;
		}
	}

	tmp1 = mem + pos, pos += tv_dim * tv_dim * nFile;
	n = tv_dim * tv_dim;
	for (i = 0; i < nGauss; ++i) {
		for (j = 0; j < nFile; j++) {
			idx = j * n;
			for (k = 0; k < n; ++k) {
				sum = Exx[idx + k] * N[j][i];
				tmp1[idx + k] = (float)sum;	
				if(DEBUG_CHECK_LIMIT(sum))
					sum = sum;
// 				tmp1[idx + k] = Exx[idx + k] * N[j][i];	// tmp = bsxfun(@times, Exx, reshape(N1(:, mix),[1 1 len]));
			}
		}

		idx = i * n;
		for (j = 0, sum = 0; j < n; ++j, sum = 0) {
			for (k = 0; k < nFile; k++)
				sum += tmp1[k * n + j];
			LU[idx + j] += (float)sum;					// LU{mix} = LU{mix} + sum(tmp, 3);
			if(DEBUG_CHECK_LIMIT(sum))
				sum = sum;
		}
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
			memcpy(&tmp[j * nDim], &RU[j * n + idx], sizeof(float) * nDim);

		memcpy(pLU_inv, pLU, sizeof(float) * tv_dim * tv_dim);
		inv(pLU_inv, tv_dim);
		if (i == 137)
			i=i;

		for (j = 0; j < tv_dim; ++j) {
			for (k = 0, sum = 0; k < nDim; ++k, sum = 0) {
				for (l = 0; l < tv_dim; ++l)
					sum += pLU_inv[j * tv_dim + l] * tmp[l * nDim + k];
				if(DEBUG_CHECK_LIMIT(sum))
					sum = sum;
				tmp2[j * nDim + k] = (float)sum;
			}
		}

		// tmp2 copy to RU
		for (j = 0; j < tv_dim; ++j)
			memcpy(&RU[j * n + idx], &tmp2[j * nDim], sizeof(float) * nDim);
	}

	// Done. result store in T
	for (i = 0; i < tv_dim; ++i)
		memcpy(T[i], &RU[i * n], sizeof(float) * n);

	SAFE_FREE(T_invS);
	SAFE_FREE(mem);
	return 0;
}

#ifdef _HARDCODE_TEST
void LoadT(float **T, int m, int n)
{
	int i, j;
	char ctmp[64];
	FILE *fp = fopen("T.txt", "r");

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			fscanf(fp, "%s", ctmp);
			T[i][j] = (float)atof(ctmp);
		}
	}
	fclose(fp);
}
#endif

void TrainTVSpace(GMMDATA *pUbmData, float **NF, int nFile, int tv_dim, int nIter, float *T)
{
	int i, n;
	int nDim = pUbmData->m_nDim;
	int nGauss = pUbmData->m_nNumGMMGauss;
	float **N = NULL, **F = NULL, **ppT = NULL;
	float *S = &pUbmData->m_pfMemPool[nDim * nGauss];
	float sum_sigma = 0;

	n = nDim * nGauss;
	for(i = 0; i < n; ++i)
		sum_sigma += S[i];
	sum_sigma /= 1000;

	N = (float **)malloc(sizeof(float *) * nFile);
	F = (float **)malloc(sizeof(float *) * nFile);

	for (i = 0; i < nFile; ++i) {
		N[i] = &NF[i][0];
		F[i] = &NF[i][nGauss];
	}

	n = tv_dim * nDim * nGauss;
	ppT = (float **)malloc(sizeof(float *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		ppT[i] = &T[i * nDim * nGauss];
	
#ifdef _HARDCODE_TEST
	LoadT(ppT, tv_dim, nDim * nGauss);  /*** HardCode randn for test ***/
 	for (i = 0; i < n; ++i) 
 		T[i] *= sum_sigma;	// T = T * sum(S) * 0.001;
#else
	for (i = 0; i < n; ++i) 
		T[i] = ((float)rand() / RAND_MAX) * sum_sigma;
#endif
	
	for (i = 0; i < nIter; ++i)
		expectation_tv(ppT, N, F, S, nFile, tv_dim, nGauss, nDim);

	SAFE_FREE(N);
	SAFE_FREE(F);
	SAFE_FREE(ppT);
}

void ExtractIVector(GMMDATA *pUbmData, float *NF, int tv_dim, float *T, float *iv)
{
	int i, j, k, l, n, size, pos = 0;
	int nDim = pUbmData->m_nDim;
	int nGauss = pUbmData->m_nNumGMMGauss;
	double sum = 0;
	float *mempool = NULL;
	float *N = NULL, *F = NULL, **ppT = NULL;
	float *L = NULL, *Cxx = NULL, *B = NULL, *tmp = NULL;
	float *T_invS_mem = NULL, **T_invS = NULL;
	float *S = &pUbmData->m_pfMemPool[nDim * nGauss];
	
	n = nDim * nGauss;

	N = NF;
	F = NF + nGauss;
	ppT = (float **)malloc(sizeof(float *) * tv_dim);

	for (i = 0; i < tv_dim; ++i)
		ppT[i] = &T[i * n];	// T[tv_dim][nDim * nGauss]

	// size = T_invS_mem + L + Cxx + B + tmp
	size = (tv_dim * n) + (tv_dim * tv_dim) + (tv_dim * tv_dim) + tv_dim + (tv_dim * n);
	mempool = (float *)malloc(sizeof(float) * size);

	T_invS_mem = mempool, pos += tv_dim * n;
	L   = mempool + pos, pos += tv_dim * tv_dim;
	Cxx = mempool + pos, pos += tv_dim * tv_dim;
	B   = mempool + pos, pos += tv_dim;
	tmp = mempool + pos, pos += tv_dim * n;

	T_invS = (float **)malloc(sizeof(float *) * tv_dim);

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
			L[j * tv_dim + k] = (float)sum;
		}
	}
	pinv(L, tv_dim, tv_dim, Cxx);			// Cxx = pinv(L)	

	for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
		for (k = 0; k < n; ++k)
			sum += T_invS[j][k] * F[k];	//  B = T_invS * F;
		B[j] = (float)sum;
	}

	for (j = 0, sum = 0; j < tv_dim; ++j, sum = 0) {
		for (k = 0; k < tv_dim; ++k)
			sum += Cxx[j * tv_dim + k] * B[k];
		iv[j] = (float)sum;			//  x = pinv(L) * B;
	}

	SAFE_FREE(mempool);
	SAFE_FREE(ppT);	
	SAFE_FREE(T_invS);
}
