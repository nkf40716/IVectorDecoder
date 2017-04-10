#include <stdlib.h> 
#include <stdio.h>
#include "base_types.h"
#include "IvDecoder.h"

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

// calculate the cofactor
static void GetMinor(float *src, float *dst, int row, int col, int n)
{
    int nCol=0, nRow=0;
 
    for(int i = 0; i < n; i++) 
	{
        if(i != row) 
		{
            nCol = 0;
            for(int j = 0; j < n; j++) 
			{           
                if( j != col ) 
				{
                    dst[nRow * (n - 1) + nCol] = src[i * n + j];
                    nCol++;
                }
            }
            nRow++;
        }
    }
}
 
// Calculate the determinant, n >= 0
static double determinant(float *x, int n)
{
    // stop the recursion
    if(n == 1) return x[0];
 
    double d = 0; 
    float *m = (float *)malloc(sizeof(float) * (n-1) * (n-1));
 
    for(int i = 0; i < n; i++ )
    {
        // get minor of element (0,i)
        GetMinor(x, m, 0, i, n);
		//d += pow(-1.0, i) * x[i] * determinant(minor, n-1);
        d += (i % 2 == 1 ? -1.0 : 1.0) * x[i] * determinant(m, n-1);
        
    }
	
    free(m);
    return d;
}


// Input:  X == two dimensional array matrix
//         n == order
// Output: Y
void inv(float *X, int n, float *Y)
{
    // calculate the determinant
    double d = 1.0 / determinant(X, n);
 
    float *minor = (float *)malloc(sizeof(float) * (n-1) * (n-1));
 
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
        {
            // get the co-factor (matrix) of A(j,i)
            GetMinor(X, minor, j, i, n);
			double mul = d * determinant(minor, n-1);	// prevent float overflow. 2017.03.30
			if (mul > FLT_MAX) 
				mul = FLT_MAX;
			else if (mul < -FLT_MAX + 0.001)
				mul = -FLT_MAX + 0.001;
			Y[i * n + j] = (float)mul;
//          Y[i * n + j] = (float)(d * determinant(minor, n-1));
			
            if((i+j) % 2 == 1)
                Y[i * n + j] = -Y[i * n + j];
        }
    }
    free(minor);
}



// Input:  X == two dimensional array matrix
//         n == order
// Output: X
void inv(float *X, int n)
{
	int i, j, k;
	static float a[100][200], t;

	if (n > 100) return;

    for(i = 0; i < n; i++)
		memcpy(a[i], &X[i * n], sizeof(float) * n);
   
	for(i = 0; i < n; i++)
	{
		for(j = n ; j < 2 * n ; j++)
		{
			if(i == j - n)
				a[i][j] = 1;
			else
				a[i][j] = 0;
		}
	}

	for(i = 0; i < n; i++)
	{
		t = a[i][i];
		for(j = i; j < 2 * n; j++) {
			a[i][j] = a[i][j] / t;
			CHECK_FLT_LIMIT(a[i][j]);
		}
		for(j = 0; j < n; j++)
		{
			if(i != j)
			{
				t = a[j][i];
				for(k = 0; k < 2 * n; k++) {
					a[j][k] = a[j][k] - t * a[i][k];
					CHECK_FLT_LIMIT(a[j][k]);
				}
			}
		}
	}

    for(i = 0; i < n; i++)
		memcpy(&X[i * n], &a[i][n], sizeof(float) * n);
}