#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "IvDecoder.h"
#include "FloatModel.h"
#include "FeaturePool.h"

#define k_nTvDim		100
#define k_nTrainIter	5

int LoadBin(char fname[], char bin[])
{
	int size;
	FILE *fp;

	if((fp = fopen(fname, "rb")) == NULL){
		printf("ERROR : fail to open binary file [%s]\n", fname);
		exit(-1);
	}

	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if(bin){
		if(fread(bin, 1, size, fp) < (size_t)size){
			printf("ERROR : fail to read binary file [%s]\n", fname);
			exit(-1);
		}
	}

	fclose(fp);
	return size;
}

GMMDATA *InitGmmData(char *lpszGMM)
{
	int i, j, n, pos, nDim, nGauss;
	GMMDATA *pGMMDATA = (GMMDATA *)malloc(sizeof(GMMDATA));

	n = LoadBin(lpszGMM, NULL);
	if (n <= 0) return NULL;
	char *lpbyGMM = (char *)malloc(sizeof(char) * n);
	LoadBin(lpszGMM, lpbyGMM);

	float *m, *v, *state;
	HANDLE hUBM = InitModelTable((float *)lpbyGMM);
 	nDim = GetDim(hUBM);
	state = GetState(hUBM, 1);
	nGauss = (int)state[0];

	float *gmm_mempool = (float *)malloc((sizeof(float) * nGauss * nDim) * 2 + (sizeof(float) * nGauss));
	float **mu = (float **)malloc(sizeof(float *) * nGauss);
	float **sigma = (float **)malloc(sizeof(float *) * nGauss);
	float *logw = &gmm_mempool[nGauss * nDim * 2];
	pos = nGauss * nDim;

	for(i = 0; i < nGauss; ++i){
		mu[i] = &gmm_mempool[i * nDim];
 		sigma[i] = &gmm_mempool[pos + i * nDim];		

		logw[i] = state[1 + (int)state[0] + i];			// logweight = state[1 + (int)state[0] + mixtureIndex]
		m = &state[1 + 2 * (int)state[0] + i * nDim * 2];
		v = m + nDim;
		for(j = 0; j < nDim; ++j){
			mu[i][j] = m[j];        // means
			sigma[i][j] = v[j];     // variances
		}
	}	
	ReleaseModelTable(hUBM);
	free(lpbyGMM);

	pGMMDATA->m_nDim = nDim;
	pGMMDATA->m_nNumGMMGauss = nGauss;
	pGMMDATA->m_pfMemPool = gmm_mempool;
	pGMMDATA->m_ppfMeans = mu;
	pGMMDATA->m_ppfSigma = sigma;
	pGMMDATA->m_pfLogW = logw;
	
	return pGMMDATA;
}

int ReleaseGmmData(GMMDATA *pGMMDATA)
{
	if (!pGMMDATA) return -1;
	SAFE_FREE(pGMMDATA->m_ppfMeans);
	SAFE_FREE(pGMMDATA->m_ppfSigma);
	SAFE_FREE(pGMMDATA->m_pfMemPool);
	SAFE_FREE(pGMMDATA);
	return 0;
}

static int DumpBin(char *lpszFile, char *lpbyBin, int nBinSize)
{
	FILE *fp;

	if (lpbyBin == NULL || nBinSize <= 0)
		return -1;

	if ((fp = fopen((char *)lpszFile, "wb")) == NULL)
		return -1;

	if ((int)fwrite(lpbyBin, 1, nBinSize, fp) != nBinSize) {
		fclose(fp);
		unlink(lpszFile);
		return -1;
	}
	fclose(fp);

	return nBinSize;
}

static int DumpIvBin(char *lpszFile, char *lpbyBin, int nBinSize, int tv_dim, int nUttr)
{
	FILE *fp;

	if (lpbyBin == NULL || nBinSize <= 0)
		return -1;

	if ((fp = fopen((char *)lpszFile, "wb")) == NULL)
		return -1;

	fwrite(&tv_dim, 1, sizeof(int), fp);
	fwrite(&nUttr, 1, sizeof(int), fp);

	if ((int)fwrite(lpbyBin, 1, nBinSize, fp) != nBinSize) {
		fclose(fp);
		unlink(lpszFile);
		return -1;
	}
	fclose(fp);

	return nBinSize + sizeof(int) + sizeof(int);
}

int main(int argc, char* argv[])
{
	if (argc < 4) {
		printf("Usage: %s <UBM> <FeaturePool> [testgmm file]\n", argv[0]);
		exit(-1);
	}

	/*
	int i, j, n, r;
	FILE *fp = fopen(argv[3], "r");
	if (fp == NULL) {
		printf("ERROR: open file failed. [%s]\n", argv[1]);
		exit(-1);
	}
	
	const int nMixtures = 32;
	const int nDim      = 13;
	const int nFrames   = 1000;

	char ctmp[64];
	GMMDATA *pUbmData = (GMMDATA *)malloc(sizeof(GMMDATA));
	float *data = (float *)malloc(sizeof(float) * nFrames * nDim);
	float *gmm_mempool = (float *)malloc((sizeof(float) * nMixtures * nDim) * 2 + (sizeof(float) * nMixtures));
	float **mu = (float **)malloc(sizeof(float *) * nMixtures);
	float **sigma = (float **)malloc(sizeof(float *) * nMixtures);
	float *logw = &gmm_mempool[nMixtures * nDim * 2];
	int pos = nMixtures * nDim;

	for (i = 0; i < nMixtures; ++i) {
		mu[i] = &gmm_mempool[i * nDim];
		sigma[i] = &gmm_mempool[pos + i * nDim];
	}

	for (i = 0; i < nMixtures; ++i) {
		fscanf(fp, "%s", ctmp);
		logw[i] = (float)log(atof(ctmp));
	}
	
	for (i = 0; i < nDim; ++i) {
		for (j = 0; j < nMixtures; ++j) {
			fscanf(fp, "%s", ctmp);
			mu[j][i] = (float)atof(ctmp);
		}
	}
	
	for (i = 0; i < nDim; ++i) {
		for (j = 0; j < nMixtures; ++j) {
			fscanf(fp, "%s", ctmp);
			sigma[j][i] = (float)atof(ctmp);
		}
	}

	n = 0;
	for (i = 0; i < nFrames; ++i) {
		for (j = 0; j < nDim; ++j)
			data[n++] = (float)(i + j + 2) / 100;
	}
	fclose(fp);

	pUbmData->m_nDim = nDim;
	pUbmData->m_nNumGMMGauss = nMixtures;
	pUbmData->m_pfMemPool = gmm_mempool;
	pUbmData->m_ppfMeans = mu;
	pUbmData->m_ppfSigma = sigma;
	pUbmData->m_pfLogW = logw;

	const int nUttr = 1;
	const int k_nTvDim = 100;
	const int k_nTrainIter = 5;
	
	float *T = (float *)malloc(sizeof(float) * tv_dim * nMixtures * nDim);
	float *iv_mem = (float *)malloc(sizeof(float) * tv_dim * nUttr);
	float **iv = (float **)malloc(sizeof(float *) * nUttr);
	float **NF = (float **)malloc(sizeof(float *) * nUttr);
	for (i = 0; i < nUttr; ++i) {
		NF[i] = (float *)malloc(sizeof(float) * (nMixtures + nMixtures * nDim));
		r = compute_bw_stats_float(data, 1000, pUbmData, NF[i]);
	}

 	TrainTVSpace(pUbmData, NF, nUttr, tv_dim, nIter, T);

	for (i = 0; i < nUttr; ++i)
	{
		iv[i] = &iv_mem[i * tv_dim];
		ExtractIVector(pUbmData, NF[i], tv_dim, T, iv[i]);
	}
	
	for (i = 0; i < nUttr; ++i)
		SAFE_FREE(NF[i]);
	SAFE_FREE(NF);	
	SAFE_FREE(iv_mem);
	SAFE_FREE(iv);
	SAFE_FREE(T);
	SAFE_FREE(pUbmData);
	SAFE_FREE(data);
	SAFE_FREE(gmm_mempool);
	SAFE_FREE(mu);
	SAFE_FREE(sigma);
	return 0;
	*/

	// Test LoadUBM Code

	int i, n, r;
	class FeaturePool *poFPool = new FeaturePool(argv[2]);
	int nUttr = poFPool->GetNumUtterance();
	if (nUttr <= 0) 
	{
		printf("Init feature pool failed.\n");
		exit(-1);
	}
	int nDim = poFPool->GetDim();	
	int nMaxVector = 0;
	for (i = 0; i < nUttr; ++i) 
	{	
		n = poFPool->GetNumVector(i);
		if (n > nMaxVector) nMaxVector = n;
	}
	
	char *fea = (char *)malloc(sizeof(char) * nMaxVector * nDim);
	
	GMMDATA *pUbmData = InitGmmData(argv[1]);
	int nMixtures = pUbmData->m_nNumGMMGauss;
	const int tv_dim = k_nTvDim;
	const int nIter = k_nTrainIter;

	float *T = (float *)malloc(sizeof(float) * tv_dim * nMixtures * nDim);
	float *iv_mem = (float *)malloc(sizeof(float) * tv_dim * nUttr);
	float **iv = (float **)malloc(sizeof(float *) * nUttr);
	float **NF = (float **)malloc(sizeof(float *) * nUttr);
	for (i = 0; i < nUttr; ++i) 
	{
		n = poFPool->GetVector(i, fea);
		NF[i] = (float *)malloc(sizeof(float) * (nMixtures + nMixtures * nDim));
		r = compute_bw_stats(fea, n, pUbmData, NF[i]);
	}
	SAFE_FREE(fea);

	double nStart = clock();

 	TrainTVSpace(pUbmData, NF, nUttr, tv_dim, nIter, T);

	double nEnd = clock();
	printf("Training time : %.2f secs\n", (nEnd - nStart) / CLOCKS_PER_SEC);

	nStart = clock();

	for (i = 0; i < nUttr; ++i)
	{
		iv[i] = &iv_mem[i * tv_dim];
		ExtractIVector(pUbmData, NF[i], tv_dim, T, iv[i]);
	}

	nEnd = clock();
	printf("Extract time : %.2f secs\n", (nEnd - nStart) / CLOCKS_PER_SEC);

	// save ivector file
	DumpIvBin(argv[4], (char*)iv_mem, sizeof(float) * tv_dim * nUttr, tv_dim, nUttr);


	ReleaseGmmData(pUbmData);
	for (i = 0; i < nUttr; ++i)
		SAFE_FREE(NF[i]);
	SAFE_FREE(NF);
	SAFE_FREE(iv_mem);
	SAFE_FREE(iv);
	SAFE_FREE(T);
  	SAFE_DELETE(poFPool);
	return 0;
}