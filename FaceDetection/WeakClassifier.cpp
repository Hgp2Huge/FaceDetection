#include "stdafx.h"
#include <fstream>
#include <vector>
#include <math.h>
using namespace std;
#include "FDImage.h"
#include "WeakClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"

/*
void WeakClassifier::getFeatureDescriptor(FDImage &im,float* pFea)
{
	int pointIndex[4];
	int width = im.width;int height = im.height;
	if(rh == rw*4){
		// 4x1 cells
		// |------|
		// |   1  |
		// |------|
		// |  2   |
		// |------|
		// |   3  |
		// |------|
		// |   4  |
		// |------|
		//
		pointIndex[0] = y*width+x;pointIndex[1] = y*width+x+rw;
		pointIndex[2] = y*width+x+rh/4;pointIndex[3] = y*width+x+rw+rh/4;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rh/4;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else if(rh == rw/4){
		// 1x4 cells
		// |------|------|------|------|
		// |   1  |	 2   |	 3  |	4  |
		// |------|------|------|------|
		pointIndex[0] = y*width+x;pointIndex[1] = y*width+x+rw/4;
		pointIndex[2] = y*width+x+rh;pointIndex[3] = y*width+x+rw/4+rh;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rw/4;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else if(rh == rw/2){
		// 2x2 cells
		// |------|------|
		// |   1  |  2   |
		// |------|------|
		// |   3  |  4   |
		// |------|------|
		pointIndex[0] = y*width+x;pointIndex[1] = y*width+x+rw/2;
		pointIndex[2] = y*width+x+rh/2;pointIndex[3] = y*width+x+rw/2+rh/2;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+8,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		pointIndex[0] -= rw/2;pointIndex[1] -= rw/2;pointIndex[2] -= rw/2;pointIndex[3] -= rw/2;
		pointIndex[0] += rh/2;pointIndex[1] += rh/2;pointIndex[2] += rh/2;pointIndex[3] += rh/2;
		getRectFeatureDescriptor(im,pFea+16,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+24,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
	}
}


void WeakClassifier::getRectFeatureDescriptor(FDImage &im,float* pFea,int p1,int p2,int p3,int p4)
{
	double* a1 = (im.m_integralIamge[p1].f);
	double* a2 = (im.m_integralIamge[p2].f);
	double* a3 = (im.m_integralIamge[p3].f);
	double* a4 = (im.m_integralIamge[p4].f);
	pFea[0] = (a1[0] - a2[0] + a3[0] - a4[0]);
	pFea[1] = (a1[1] - a2[1] + a3[1] - a4[1]);
	pFea[2] = (a1[2] - a2[2] + a3[2] - a4[2]);
	pFea[3] = (a1[3] - a2[3] + a3[3] - a4[3]);
	pFea[4] = (a1[4] - a2[4] + a3[4] - a4[4]);
	pFea[5] = (a1[5] - a2[5] + a3[5] - a4[5]);
	pFea[6] = (a1[6] - a2[6] + a3[6] - a4[6]);
	pFea[7] = (a1[7] - a2[7] + a3[7] - a4[7]);
}
*/
double WeakClassifier::Apply( FDImage &im){
	double pFea[32],ret;
	getFeatureDescriptor(im,pFea,x,y,rh,rw);
	double yfea = 0.0;
	yfea = 1*weights[0];
//	trainlogfile << "  WeakApply::patch :"  << x <<' ' << y << rh << rw << "  ";
//	for(int j =0; j < 33;j++){
//		trainlogfile << weights[j] << "  ";
	//}

	for(int j =0; j < 32;j++){
		yfea += pFea[j]*weights[j+1];
//		trainlogfile << "  WeakApply::pFea :"  << pFea[j] << "  ";
	}
	//trainlogfile << endl;

	//trainlogfile << "  WeakApply::yfea :"  << yfea << "  ";
	ret = sigmoid(yfea);
	return ret;
}

void WeakClassifier::WriteToFile(ofstream& f)
{
	f<< x <<' ' << y <<' '  << rh <<' '  << rw ;
	for(int i=0;i<33;i++) 
		f << ' ' << weights[i];
	f<<endl;
}

void WeakClassifier::ReadFromFile(ifstream& f)
{
	f >>  x >> y >>  rh >> rw ;
	for(int i=0;i<33;i++) 
		f >> weights[i];
	f.ignore(256,'\n');
}