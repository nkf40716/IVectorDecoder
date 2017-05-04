#include <math.h>
#include "IvDecoder.h"

// void transpose(float *src, float *dst, int w, int h)
// {
// 	for (int x = 0; x < w; x++) {
// 		for (int y = 0; y < h; y++)
// 			*(dst + x * h + y) = *(src + y * w + x);
// 	}
// }

static void transpose(double *src, double *dst, int w, int h)
{
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++)
			*(dst + x * h + y) = *(src + y * w + x);
	}
}

static double Exp(double u)
{
	#define kNumApproximate		32768
	#define kMaxExponentDiff	(16.0)

	static int bFlagTableExist = 0;
	static double k_lpdExpTable[kNumApproximate + 1];

	int i;
	double v, w;

	// one-time log reference table generation
	if (bFlagTableExist == 0){
		for (i = 0; i < kNumApproximate + 1; i++){
			k_lpdExpTable[i] = exp(-kMaxExponentDiff * i / kNumApproximate);
		}
		bFlagTableExist = 1;
	}

	if (u <= -kMaxExponentDiff)
		return 0;
	v = -u * kNumApproximate / kMaxExponentDiff;
	i = (int)v;
	w = v - i;
	return k_lpdExpTable[i] * (1.0 - w) + k_lpdExpTable[i+1] * w;
}

// compute the log probability of observations given the GMM.
// return size: nMixtures x nDataLen
double **lgmmprob(float **data, int nDim, int nDataLen, float **mu, float **sigma, float *logw, int nMixtures)
{
	int i, j, size, n = 0, idx = 0;
	double sum;
	double *mem = NULL;
	double *C = NULL, *t_mem = NULL;
	double *D1 = NULL, *D1_t = NULL;
	double **D = NULL, *D_mem = NULL, **t = NULL;
	
	D_mem = (double *)malloc(sizeof(double) * nDataLen * nMixtures);
	D = (double **)malloc(sizeof(double *) * nDataLen);
	t = (double **)malloc(sizeof(double *) * nDataLen);
	
	// size = C + t + D1 + D1_t 
	size = nMixtures + (nDataLen * nDim) + (nDim * nMixtures) + (nDim * nMixtures);
	mem = (double *)malloc(sizeof(double) * size);

	C = mem, n += nMixtures;
	t_mem = mem + n, n += nDataLen * nDim;

	for (i = 0; i < nDataLen; ++i) {
// 		D[i] = (double *)malloc(sizeof(double) * nMixtures);
		D[i] = &D_mem[i * nMixtures];
		t[i] = &t_mem[i * nDim];
	}
	
	// C = sum(mu.*mu./sigma) + sum(log(sigma));
	for (i = 0; i < nMixtures; ++i)	
	{
		double sumlogsig = 0;
		sum = 0;
		for (j = 0; j < nDim; ++j) {
			sum += (mu[i][j] * mu[i][j]) / sigma[i][j];
			sumlogsig += log(sigma[i][j]);
		}
		C[i] = sum + sumlogsig;
	}

	// D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);

	D1 = mem + n, n += nDim * nMixtures;
	D1_t = mem + n, n += nDim * nMixtures;

	for (i = 0; i < nMixtures; ++i)
	{
		for (j = 0; j < nDim; ++j)
			D1[i * nDim + j] = 1 / sigma[i][j];
	}
	transpose(D1, D1_t, nDim, nMixtures);			// D1_t = (1./sigma)'

	for (i = 0; i < nDataLen; ++i)
	{
		for (j = 0; j < nDim; ++j)
			t[i][j] = data[i][j] * data[i][j];		// t = (data .* data)
	}

	for (i = 0; i < nDataLen; ++i) 
	{	
		for (j = 0; j < nMixtures; ++j)
		{
			sum = 0;
			for (int x = 0; x < nDim; ++x)
				sum += D1_t[x * nMixtures + j] * t[i][x];

			D[i][j] = sum;
		}
	}	// D = (1./sigma)' * (data .* data);

	for (i = 0; i < nMixtures; ++i)
	{
		for (j = 0; j < nDim; ++j)
			D1[i * nDim + j] = mu[i][j] / sigma[i][j];
	}
	transpose(D1, D1_t, nDim, nMixtures);			// D1_t = (mu./sigma)'

	double para = nDim * 1.837877066409345;			// log(2 * pi) = 1.837877066409345
	for (i = 0; i < nDataLen; ++i) 
	{	
		for (j = 0; j < nMixtures; ++j)
		{
			sum = 0;
			for (int x = 0; x < nDim; ++x)
				sum += D1_t[x * nMixtures + j] * data[i][x];

			D[i][j] += ((-2 * sum) + para);
		}
	}		

// 	logprob = -0.5 * (bsxfun(@plus, C',  D));
// 	logprob = bsxfun(@plus, logprob, log(w));
	for (i = 0; i < nMixtures; ++i) {
		for (j = 0; j < nDataLen; ++j)
			D[j][i] = -0.5 * (C[i] + D[j][i]) + logw[i];
	}

	SAFE_FREE(mem);
	SAFE_FREE(t);
	return D;    // D[nDataLen][nMixtures], ex: D[1000][32]
}


// compute log(sum(exp(x),dim)) while avoiding numerical underflow
// compute log(sum(exp(x))) in each column. return size: 1 x w
double *logsumexp(double **x, int w, int h)
{
	int i, j;
	double *xmax = (double *)malloc(sizeof(double) * w);
	double *y = (double *)malloc(sizeof(double) * w);
	
	// find max value in each column
	for (i = 0; i < w; ++i) {
		double fm = x[i][0];
		for (j = 1; j < h; ++j) {
			if (x[i][j] > fm) 
				fm = x[i][j];
		}
		xmax[i] = fm;
	}

	for (i = 0; i < w; ++i)	{
		double sumexp = 0;
		for (j = 0; j < h; ++j) {
			sumexp += Exp(x[i][j] - xmax[i]);
		}
		y[i] = xmax[i] + log(sumexp);
	}
// Not implement: ind = find(~isfinite(xmax));
//                if ~isempty(ind) y(ind) = xmax(ind);
	
	SAFE_FREE(xmax);
	return y;
}

// compute the posterior probability of mixtures for each frame.
// return size: nMixtures x nDataLen
double **postprob(float **data, int nDim, int nDataLen, float **mu, float **sigma, float *logw, int nMixtures, double *llk)
{
	double **post = lgmmprob(data, nDim, nDataLen, mu, sigma, logw, nMixtures);
	double *pllk  = logsumexp(post, nDataLen, nMixtures);

	for (int i = 0; i < nDataLen; ++i) {
		for (int j = 0; j < nMixtures; ++j)
			post[i][j] = Exp(post[i][j] - pllk[i]);
	}

	if (llk) llk = pllk;
	else free(pllk);

	return post;
}

// compute the sufficient statistics.
// return size: F: nDim x nMixtures, N: nMixtures
int expectation(float **data, int nDim, int nFrames, float **mu, float **sigma, float *logw, int nMixtures, double *N ,double *F)
{
	double **post = postprob(data, nDim, nFrames, mu, sigma, logw, nMixtures, NULL);
	int i, j, k;

	for (i = 0; i < nMixtures; ++i) {
		double sum = 0;
		for (j = 0; j < nFrames; ++j)
			sum += post[j][i];
		N[i] = sum;
	}

	double *post_t_mem = (double *)malloc(sizeof(double) * nFrames * nMixtures);
	double **post_t = (double **)malloc(sizeof(double *) * nMixtures);
	for (i = 0; i < nMixtures; ++i)
		post_t[i] = &post_t_mem[i * nFrames];

	for (i = 0; i < nFrames; ++i) {
		for (j = 0; j < nMixtures; ++j)
			post_t[j][i] = post[i][j];
	}
	
	memset(F, 0, sizeof(double) * nDim * nMixtures);
	for (i = 0; i < nMixtures; ++i) {
		for (j = 0; j < nDim; ++j) {
			for(k = 0; k < nFrames; ++k)
				F[i * nDim + j] += data[k][j] * post_t[i][k];	// F = data * post';
		}
	}

// 	for (i = 0; i < nFrames; ++i) {free(post[i]);}
	SAFE_FREE(post[0]);
	SAFE_FREE(post);
	SAFE_FREE(post_t_mem);
	SAFE_FREE(post_t);

	return 0;
}

// % Outputs: pointer to [N; F]
// %   - N			  : mixture occupation counts (responsibilities) 
// %   - F            : centered first order stats
int compute_bw_stats(char *fea, int nNumFea, GMMDATA *pGMMDATA, double *NF)
{
	int i, j, n;
	int nDim = pGMMDATA->m_nDim;
	int nGauss = pGMMDATA->m_nNumGMMGauss;
	float **mu = pGMMDATA->m_ppfMeans;
	float **sigma = pGMMDATA->m_ppfSigma;
	float *logw = pGMMDATA->m_pfLogW;
	float *fea_mem = NULL, **ppfea = NULL;
	double *pN = NULL, *pF = NULL; 

	n = nNumFea * nDim;
	fea_mem = (float *)malloc(sizeof(float) * n);
	ppfea = (float **)malloc(sizeof(float *) * nNumFea);
	
	// float ver
// 	for (i = 0; i < n; ++i)
// 		fea_mem[i] = (float)fea[i] / 128;
// 	for (i = 0; i < nNumFea; ++i)
// 		ppfea[i] = &fea_mem[i * nDim];

	// char ver
	for (i = 0; i < n; ++i)
		fea_mem[i] = (float)fea[i];
	for (i = 0; i < nNumFea; ++i)
		ppfea[i] = &fea_mem[i * nDim];

	pN = NF;
	pF = NF + nGauss;
 	expectation(ppfea, nDim, nNumFea, mu, sigma, logw, nGauss, pN, pF);

	n = nDim * nGauss;
	for (i = 0, j = 0; i < n; ++i, ++j)
		pF[i] -= pN[i / nDim] * mu[i / nDim][j % nDim];		// centered first order stats

	SAFE_FREE(fea_mem);
	SAFE_FREE(ppfea);
	return n;
}

// Debug¥Î
int compute_bw_stats_float(float *fea, int nNumFea, GMMDATA *pGMMDATA, double *NF)
{
	int i, j, n;
	int nDim = pGMMDATA->m_nDim;
	int nGauss = pGMMDATA->m_nNumGMMGauss;
	double *pN = NULL, *pF = NULL;
	float **mu = pGMMDATA->m_ppfMeans;
	float **sigma = pGMMDATA->m_ppfSigma;
	float *logw = pGMMDATA->m_pfLogW;
	float **ppfea = (float **)malloc(sizeof(float *) * nNumFea);

	for (i = 0; i < nNumFea; ++i)
		ppfea[i] = &fea[i * nDim];
	
	pN = NF;
	pF = NF + nGauss;
 	expectation(ppfea, nDim, nNumFea, mu, sigma, logw, nGauss, pN, pF);

	n = nDim * nGauss;
	for (i = 0, j = 0; i < n; ++i, ++j)
		pF[i] -= pN[i / nDim] * mu[i / nDim][j % nDim];		// centered first order stats

	SAFE_FREE(ppfea);
	return n;
}