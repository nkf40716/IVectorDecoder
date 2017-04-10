#ifndef __FLOAT_MODEL_H__
#define __FLOAT_MODEL_H__


#ifdef __cplusplus
extern "C"{
#endif

HANDLE InitModelTable(float model[]);
int ReleaseModelTable(HANDLE hFloatModel);
float *GetWordModel(HANDLE hFloatModel, int modelID);
float *GetState(HANDLE hFloatModel, int stateID);
int GetModelStateList(HANDLE hFloatModel, int modelID, int stateIdList[]);
int GetNumTotModel(HANDLE hFloatModel);
int GetNumTotState(HANDLE hFloatModel);
int GetNumTotGauss(HANDLE hFloatModel);
int GetModelSize(HANDLE hFloatModel);
int GetDim(HANDLE hFloatModel);
int GetGaussID(HANDLE hFloatModel, int stateID, int mixtureIndex);
int CopyWholeModel(HANDLE hFloatModel, float model[]);
void UpdateGauss(HANDLE hFloatModel, int stateID, int mixtureIndex, float mean[], float var[], int dim);
void UpdateGaussGamma(HANDLE hFloatModel, int stateID, int mixtureIndex, float gamma);
void UpdateGaussLogWeight(HANDLE hFloatModel, int stateID, int mixtureIndex, float logWeight);
double StateScore(HANDLE hFloatModel, int stateID, int x[], int dim, int *nMaxIndex, int *nSecIndex, int *nThriIndex, int *nFourIndex, double lpdGamma[]);
double LikelihoodScore(HANDLE hFloatModel, int stateID, int x[], int dim);
int StateScoreKeepInfo(HANDLE hFloatModel, int stateID, float x[], int dim, double *likelihood, double gamma[]);



#ifdef __cplusplus
}
#endif


#endif // __FLOAT_MODEL_H__

