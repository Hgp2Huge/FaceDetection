#include "Global.h"

struct WeakClassifier{
	int x,y;
	int rw,rh;
	double weights[33];

	double Apply(FDImage &im);
	//void  getFeatureDescriptor(FDImage &im,float * pFea);
//	void  getRectFeatureDescriptor(FDImage &im,float* pFea,int p1,int p2,int p3,int p4);
	void WriteToFile(ofstream& f);
	void ReadFromFile(ifstream& f);
};