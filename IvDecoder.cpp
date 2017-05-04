#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "IvDecoder.h"
#include "FloatModel.h"
#include "FeaturePool.h"

#define SAFE_DELETE_ARR(x)	if ((x) != NULL) {delete [] (x); x = NULL;}
#define PRINT(STH)			printf("%s\n", #STH)
#define k_nTvDim			100
#define k_nTrainIter		5

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

int LoadTV(char fname[], double tv[], int nTvSize)
{
	int n = LoadBin(fname, NULL);
	if ( n != (nTvSize << 2) ) {
		PRINT(Load TV failed ! Size not match);
		exit(-1);
	}

	n /= sizeof(float);
	float *T = new float [n];
	LoadBin(fname, (char *)T);

	for (int i = 0; i < n; ++i)
		tv[i] = T[i];
		
	SAFE_DELETE_ARR(T);
	return 0;
}

GMMDATA *InitGmmData(char *lpszGMM)
{
	int i, j, n, pos, nDim, nGauss;
	GMMDATA *pGMMDATA = NULL;

	n = LoadBin(lpszGMM, NULL);
	char *lpbyGMM = (char *)malloc(sizeof(char) * n);
	LoadBin(lpszGMM, lpbyGMM);

	float *m, *v, *state;
	HANDLE hUBM = InitModelTable((float *)lpbyGMM);
	if (!hUBM) {
		printf("ERROR : Init Model Table failed !\n");
		SAFE_FREE(lpbyGMM);
		exit(-1);
	}
 	nDim = GetDim(hUBM);
	state = GetState(hUBM, 1);
	nGauss = (int)state[0];

	pGMMDATA = (GMMDATA *)malloc(sizeof(GMMDATA));
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
	SAFE_FREE(lpbyGMM);

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

static inline PrintUsage()
{
	PRINT(-tv for Training Total Variability or -iv for extract I-vector.);
	PRINT(Usage: IvDecoder.exe -tv <UBM> <FeaturePool> <out>);
	PRINT(or\n\tIvDecoder.exe -iv <UBM> <FeaturePool> <TV> <TrainUttr> <out>);
}

int main(int argc, char* argv[])
{
	if (argc < 5) {
		PrintUsage();
		exit(-1);
	}
	
	BOOL bTrainTv = FALSE;
	LoadBin(argv[2], NULL);
	if (strcmp(argv[1], "-tv") == 0) 
	{
		bTrainTv = TRUE;
	}
	else if (strcmp(argv[1], "-iv") == 0) 
	{
		if (argc < 7) 
		{
			PrintUsage();
			exit(-1);
		}
		LoadBin(argv[4], NULL);
	}
	else 
	{
		printf("Error: Illegal Parameter [%s]\n", argv[1]);
		PrintUsage();
		exit(-1);
	}
		
	int i, j, n, r;
	class FeaturePool *poFPool = new FeaturePool(argv[3]);
	int nUttr = poFPool->GetNumUtterance();
	if (nUttr <= 0) 
	{
		PRINT(Init feature pool failed !);
		SAFE_DELETE(poFPool);
		exit(-1);
	}

	int nDim = poFPool->GetDim();	
	int nMaxVector = poFPool->GetNumVector(0);
	for (i = 1; i < nUttr; ++i) 
	{	
		n = poFPool->GetNumVector(i);
		if (n > nMaxVector) nMaxVector = n;
	}
		
	GMMDATA *pUbmData = InitGmmData(argv[2]);
	const int nMixtures = pUbmData->m_nNumGMMGauss;
	const int tv_dim = k_nTvDim;
	const int nIter = k_nTrainIter;
	const int nNFSize = nMixtures + nMixtures * nDim;
	const int nTvSize = tv_dim * nMixtures * nDim;

	char *fea = new char [nMaxVector * nDim];
	double nStart, nEnd;
	double *T = NULL;
	double **NF = new double* [2];
	NF[0] = new double [nNFSize], NF[1] = new double [nNFSize];

	memset(NF[0], 0, sizeof(double) * nNFSize);	
	nStart = clock();

	if (bTrainTv == FALSE) nUttr = atoi(argv[5]);
	for (i = 0; i < nUttr; ++i) 
	{
		n = poFPool->GetVector(i, fea);
		r = compute_bw_stats(fea, n, pUbmData, NF[1]);	
		for (j = 0; j < nNFSize; ++j)
			NF[0][j] += NF[1][j];
	}
	
	nEnd = clock();
	SAFE_DELETE_ARR(fea);
	printf("compute_bw_stats time : %.2f secs\n", (nEnd - nStart) / CLOCKS_PER_SEC);
	
	if (bTrainTv == TRUE)		
	{
		T = new double [nTvSize];
		float *fT = new float [nTvSize];

		nStart = clock();
		TrainTVSpace(pUbmData, NF, 1, tv_dim, nIter, T);
		nEnd = clock();
		printf("Training TV time : %.2f secs\n", (nEnd - nStart) / CLOCKS_PER_SEC);

		for (i = 0; i < nTvSize; ++i)
			fT[i] = (float)T[i];
		
		printf("Dump TV : %s\n", argv[4]);
		if (DumpBin(argv[4], (char *)fT, sizeof(float) * nTvSize) <= 0)
			PRINT(Dump TV failed !);

		SAFE_DELETE_ARR(fT);
	}
	else
	{
		T = new double [nTvSize];
		LoadTV(argv[4], T, nTvSize);

		double *iv = new double [tv_dim];
		float *fiv = new float [tv_dim];
		
		nStart = clock();
		ExtractIVector(pUbmData, NF[0], tv_dim, T, iv);	
		nEnd = clock();
		printf("Extract time : %.2f secs\n", (nEnd - nStart) / CLOCKS_PER_SEC);

		for (i = 0; i < tv_dim; ++i)
			fiv[i] = (float)iv[i];

		printf("Dump i-vector : %s\n", argv[6]);
		if (DumpBin(argv[6], (char *)fiv, sizeof(float) * tv_dim) <= 0)
			PRINT(Dump i-vector failed !);
		
		SAFE_DELETE_ARR(iv);
		SAFE_DELETE_ARR(fiv);
	}

	ReleaseGmmData(pUbmData);
	SAFE_DELETE_ARR(NF[0]);
	SAFE_DELETE_ARR(NF[1]);
	SAFE_DELETE_ARR(NF);
	SAFE_DELETE_ARR(T);
  	SAFE_DELETE(poFPool);
	return 0;
}