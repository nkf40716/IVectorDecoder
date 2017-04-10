#ifndef __FeaturePool_h__
#define __FeaturePool_h__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "base_types.h"

#ifdef _WIN32
typedef __int64	int64_t;
#else
#include <unistd.h>
#endif

enum {
	kVER_UNKNOWN = 0,
	kVER_FP,
	kVER_FPX_FixRefSize,
	kVER_FPX_VarRefSize,
	kVER_V4FLOAT,
	kVER_V5CHAR,
	kVER_V5SHORT,
	kVER_V5FLOAT,
	kVER_MAX
};

class FeaturePool {
public :
	FeaturePool(char *pcFeatureFile);
	~FeaturePool();

	int GetNumUtterance() {return m_nNumUtterance;}
	int GetDim() {return m_nDim;}

	//Get reference of an utterance; return ref. length + 1. Doesn't copy reference if lpszRef = NULL
	// return -1 if failed
	int GetReference(int nUtteranceID, char *lpszRef);
	int GetReference(int nUtteranceID, UNICODE *lpwcRef);
	//Get number of vectors of an utterance, return -1 if failed
	int GetNumVector(int nUtteranceID);
	//Get all vectors of an utterance. Return real number of vector retrieved, or -1 if failed
	int GetVector(int nUtteranceID, char *lpcVectBuf);
	int GetVector(int nUtteranceID, float *lpfVectBuf);
	int GetVector(int nUtteranceID, int *lpnVectBuf);
	//Get a vector of an utterance. Return real number of vector retrieved, or -1 if failed
	int GetVector(int nUtteranceID, int nVectID, char *lpcVectBuf);
	int GetVector(int nUtteranceID, int nVectID, float *lpfVectBuf);
	int GetVector(int nUtteranceID, int nVectID, int *lpnVectBuf);
	//Get a segment of vectors of an utterance. Return real number of vector retrieved, or -1 if failed
	int GetVector(int nUtteranceID, int nStartVect, int nEndVect, char *lpcVectBuf);
	int GetVector(int nUtteranceID, int nStartVect, int nEndVect, float *lpfVectBuf);
	int GetVector(int nUtteranceID, int nStartVect, int nEndVect, int *lpnVectBuf);

	int GetVersion() {return m_nVersion;}

private:
	bool LoadFPX_VarRefSize();
	bool LoadFPX_FixRefSize();
	bool LoadV4FLOAT();
	bool LoadV5CHAR();
	bool LoadV5SHORT();
	bool LoadV5FLOAT();
	bool LoadFP();
	int GetReference(int nUtteranceID, void *lpvRef);
	int GetVector(int nUtteranceID, int nStartVect, int nEndVect, void *lpvVectBuf);
	void Release();
	int m_nNumUtterance;
	int m_nDim;
	int64_t *m_pnUtteranceOffset;
	FILE *m_fpFeaFile;
	int m_nVersion;
};

#endif

