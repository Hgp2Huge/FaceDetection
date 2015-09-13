#include "stdafx.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cstdlib>
using namespace std;

#include "FDImage.h"
#include "WeakClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

typedef struct Roc{
	double fpr,tpr,thresh;
} Roc;

AdaBoostClassifier::AdaBoostClassifier()
{
}

AdaBoostClassifier::~AdaBoostClassifier()
{
	Clear();
}

void AdaBoostClassifier::Clear()
{
	count = 0;
	thresh = 0;
	weakClassifiers.clear();
	trainSet_sampled.clear();
}

AdaBoostClassifier& AdaBoostClassifier::operator=(const AdaBoostClassifier& source)
{
	InitToGivenSize(source.weakClassifiers.size());

	count = source.count;
	thresh = source.thresh;
	weakClassifiers = source.weakClassifiers;

	return *this;
}

void AdaBoostClassifier::InitToGivenSize(const int size)
{
	Clear();
	count = 0;
	thresh = 0.0;
	weakClassifiers.clear();
}

void AdaBoostClassifier::WriteToFile(ofstream& f)
{
	count = weakClassifiers.size();
	f<<count << ' ' << thresh<<endl;
	for(int i=0;i<count;i++) weakClassifiers[i].WriteToFile(f);
	f<<endl;
}

void AdaBoostClassifier::ReadFromFile(ifstream& f)
{
	Clear();
	f >> count >> thresh;f.ignore(256,'\n');
	WeakClassifier tweak;
	for(int i=0;i<count;i++){
		tweak.ReadFromFile(f);
		weakClassifiers.push_back(tweak);
	}
	f.ignore(256,'\n');
}

void AdaBoostClassifier::TrainAdaBoost(int rounds,const int nPositive,const int nNegative,double dmin,double &f_cur,double &d_cur,ofstream& fileused)
{
	Clear();
	int nSampled = gsampled;
	double* weights_cur = new double[ nPositive + nNegative];
	vector<Feature> xfea ;
	vector<int> ylabel;
	vector<double> gAUCs;
	double variance1=0.0,variance2=1.0;
	for(int k =0; k < nPositive; k++){
		weights_cur[k] = (double)1/nPositive;
	}
	for(int k =nPositive; k < (nPositive+nNegative); k++){
		weights_cur[k] = (double)1/nNegative;
	}

	vector<WeakClassifier> hk;

	trainlogfile << "#Rounds#:" << rounds << endl;
	for(int i=0;i<rounds;i++){	
		trainSet_sampled.clear();
		trainlogfile << " ..............the " << i+1 << " round.............."  << endl;
		// sample positive 
		trainlogfile << "sampling positive set" << endl;
		 fileused << endl << "round: " << i+1  << " positive set "<< endl;
		sampleTrainSet(trainset_Cur,weights_cur,nPositive,nSampled,fileused);
		// sample negative
		trainlogfile << "sampling negative set" << endl;
		fileused << endl <<" round: " << i+1  << " negative set "<< endl;
		sampleTrainSet(trainset_Cur+nPositive,weights_cur+nPositive,nNegative,nSampled,fileused);
		fileused << endl;
		trainlogfile << "sampling xfea" << endl  << flush;
		xfea.clear();
		sampleTrainFea(xfea,nSampled);
		ylabel.clear();
		for(int j = 0; j < nSampled*2; j++)	ylabel.push_back(trainset[trainSet_sampled[j]].label);
		hk.clear();
		trainlogfile << "train logistic" << endl;
		trainLogistic(xfea,ylabel,hk,nSampled*2);
	//	trainlogfile << "hk size: "  << hk.size() << endl;
		double nAUCs = 0;
		int choice=0;
		trainlogfile << "#AUCs#: " ;
		for(int j = 0; j < totalfeatures; j++){
			double tAUCs;
			weakClassifiers.push_back(hk[j]);
		//	trainlogfile << "weakclassifier : " << j << " weak's size:" << weakClassifiers.size() << endl;
			tAUCs = CalcAUCscore(nPositive,nNegative);
			if(nAUCs < tAUCs){
				choice = j;
				nAUCs = tAUCs;
			}
			weakClassifiers.pop_back();
		}
		weakClassifiers.push_back(hk[choice]);
		trainlogfile  << "choice# " << choice << ":: ";
		hk[choice].WriteToFile(trainlogfile);

	//	UpdateandNormalizeWeights();
		trainlogfile << "update weights" <<endl;
		for(int j = 0; j < (nPositive+nNegative); j++)
			weights_cur[j] = weights_cur[j]*exp(-labels[trainset_Cur[j]]*hk[choice].Apply(trainset[trainset_Cur[j]]));
		FeatureNormalization(weights_cur,nPositive);
		FeatureNormalization(weights_cur+nPositive,nNegative);
	/*	for(int j = 0; j < (nPositive+nNegative); j++)
			trainlogfile << weights_cur[j] << " " ;
		trainlogfile << endl;*/
		nAUCs = CalcAUCscore(nPositive,nNegative);
		trainlogfile << "Global AUC#: " << nAUCs << endl;
		gAUCs.push_back(nAUCs);
		//if(fabs(gAUCs - nAUCs) < 0.0001)
		//	break;
		//else
		//	gAUCs = nAUCs;
		if(i >= 5){
			variance2 = gAUCs[i] + gAUCs[i-1] +gAUCs[i-2] + gAUCs[i-3] + gAUCs[i-4]+ gAUCs[i-5];
			variance2 /= 6.0;
			variance2 = pow( gAUCs[i]-variance2,2) + pow( gAUCs[i-1]-variance2,2) +   pow( gAUCs[i-2]-variance2,2) +   pow( gAUCs[i-3]-variance2,2) +   pow(gAUCs[i-4]-variance2,2) + pow(gAUCs[i-5]-variance2,2);   
			variance2 = sqrt(variance2);
		}
		trainlogfile << "variance before: " << variance1 << " variance cur: " << variance2 <<  endl;
		if(fabs(variance2 - variance1) < 0.001)
			break;
		else if(i >= 4){
			variance1 = variance2;
			variance2 = 1.0;
		}

	}

	trainlogfile << endl << "Global AUC#:" << endl;
	for(UINT j = 0; j <gAUCs.size();j++)
		trainlogfile << gAUCs[j] << " ";
	trainlogfile << endl;

	count = weakClassifiers.size();

	trainlogfile << "Search on ROC:" <<endl;
	CalcAUCscore(dmin,f_cur,d_cur,nPositive,nNegative);
	trainlogfile << "thresh:" << thresh << "f_cur: " << f_cur << "d_cur: " << d_cur << endl; 

	delete[] weights_cur;weights_cur=NULL;
}

typedef struct sampledX{
	double x;bool label;

}sampledX;
int my_cmp(sampledX x1,sampledX  x2)
{
	return x1.x > x2.x;
}

double AdaBoostClassifier::CalcAUCscore(double dmin,double &f_cur,double &d_cur,const int nPositive,const int nNegative)
{
	if(nPositive < 0 || nNegative < 0)
		return 0;

	double FP,TP,FPpre,TPpre;
	double AreaScore = 0.0;
	double tfpre = 1.1;
	int n = nPositive+nNegative;
	//double* tfeaProbalisitic = new double[n];

	double diff,tdiff;

	//int* tfealabels = new int[n];
	//int* labelsIndex = new int[n];
//	for(int j = 0; j < n; j++){
//		tfeaProbalisitic[j] = GetValue(trainset[trainset_Cur[j]]);
//		tfealabels[j] = trainset[trainset_Cur[j]].label;
//		labelsIndex[j] = j;
		//trainlogfile << tfeaProbalisitic[j] << " " <<trainset_Cur[j] << " "  <<  trainset[trainset_Cur[j]].label<<endl;
//	}
	//trainlogfile << endl << "after sort:" << endl;
//	QuickSortDescend(tfeaProbalisitic,labelsIndex,n);

	
	vector<sampledX> tfeaProbalisitic;
	sampledX tx;
	for(int j = 0; j < n; j++){
		tx.x = GetValue(trainset[trainset_Cur[j]]);
		tx.label =trainset[trainset_Cur[j]].label ;
		tfeaProbalisitic.push_back(tx);
	
	}
	sort(tfeaProbalisitic.begin(),tfeaProbalisitic.end(),my_cmp);



	//for(int j = 0; j < n; j++){
	//	trainlogfile << tfeaProbalisitic[j] << " " << labelsIndex[j] << " "<< trainset_Cur[labelsIndex[j]]<< " " <<  trainset[trainset_Cur[labelsIndex[j]]].label <<endl;
	//}
	//trainlogfile << endl << "pro:" << endl;
	FP = TP = FPpre = TPpre = 0.0;

	int i=0;
	vector<Roc> RocCurs;
	Roc troc;
	while(i < n){
		if(tfeaProbalisitic[i].x != tfpre){
	//		trainlogfile << i << " " << labelsIndex[i] << " " << tfealabels[labelsIndex[i]] << " " << trainset_Cur[labelsIndex[i]] <<" "<< tfeaProbalisitic[i] << " " << " " << FP << " " << TP<<endl; 
			tfpre = tfeaProbalisitic[i].x;
			troc.fpr = FP/nNegative;
			troc.tpr = TP/nPositive;
			troc.thresh = tfeaProbalisitic[i].x;
			RocCurs.push_back(troc);
			rocfile << troc.fpr << " " << troc.tpr << " " << troc.thresh << endl;
		}
		if(tfeaProbalisitic[i].label == 1)
			TP++;
		else
			FP++;
		i++;
	}
	troc.fpr = FP/nNegative;troc.tpr = TP/nPositive;
	rocfile << troc.fpr << " " << troc.tpr << " " << troc.thresh << endl;
	RocCurs.push_back(troc);

	diff = 1.0;tdiff=0.0;
	int thindex;
	for(int k = RocCurs.size()-1 ; k >=0; k--){
		tdiff = fabs(dmin-RocCurs[k].tpr);
		if(tdiff < diff){
			diff = tdiff;
			thindex = k;
		}
	}
	thresh = RocCurs[thindex].thresh;
	f_cur = RocCurs[thindex].fpr;
	d_cur = RocCurs[thindex].tpr;

//	delete[] tfeaProbalisitic;
	//delete[] tfealabels;
	//delete[] labelsIndex;
	return 0.0;

}
double AdaBoostClassifier::CalcAUCscore(int nPositive,int nNegative)
{
	if(nPositive < 0 || nNegative < 0)
		return 0;
	double FP,TP,FPpre,TPpre;
	double AreaScore = 0;
	double tfpre = 1.1;
	int n = (nPositive+nNegative);
//	double* tfeaProbalisitic = new double[n];
//	int* tfealabels = new int[n];
//	int* labelsIndex = new int[n];

	//for(int j = 0; j < n; j++){
	//	tfeaProbalisitic[j] = GetValue(trainset[trainset_Cur[j]]);
	//	tfealabels[j] = labels[trainset_Cur[j]];
	//	labelsIndex[j] = j;
	////	trainlogfile << "image:" <<trainSet_sampled[j] <<" p:" << tfeaProbalisitic[j] << ' ' << "L:" << tfealabels[j] << ";  " << endl;
	//}
	vector<sampledX> tfeaProbalisitic;
	sampledX tx;
	for(int j = 0; j < n; j++){
		tx.x = GetValue(trainset[trainset_Cur[j]]);
		tx.label =trainset[trainset_Cur[j]].label ;
		tfeaProbalisitic.push_back(tx);
	//	trainlogfile << "image:" <<trainSet_sampled[j] <<" p:" << tfeaProbalisitic[j] << ' ' << "L:" << tfealabels[j] << ";  " << endl;
	}
	sort(tfeaProbalisitic.begin(),tfeaProbalisitic.end(),my_cmp);
	//QuickSortDescend(tfeaProbalisitic,labelsIndex,n);
	//for(int j = 0; j < n; j++){
	//	trainlogfile << "image:" <<trainSet_sampled[j] <<" p:" << tfeaProbalisitic[j] << ' ' << "Lindex:" << labelsIndex[j] << " L:"<< tfealabels[labelsIndex[j]]<< ";  " << endl;
	//}
	FP = TP = FPpre = TPpre = 0;

	int i=0;
	while(i < n){
		if(tfeaProbalisitic[i].x != tfpre){
			AreaScore += Trap_Area(FP,FPpre,TP,TPpre);
	//		trainlogfile << "i: " << i << " FP: " << FP << ' ' << " TP: " << TP << " Area: " <<  AreaScore << endl;
			tfpre = tfeaProbalisitic[i].x;
			FPpre = FP;
			TPpre = TP;
		}
		if(tfeaProbalisitic[i].label == 1)
			TP++;
		else
			FP++;
		i++;
	}
	AreaScore += Trap_Area(FP,FPpre,TP,TPpre);
	//trainlogfile << " Area: " <<  AreaScore << endl;
	AreaScore /= 1.0*(nPositive*nNegative);
	//trainlogfile << " Area: " <<  AreaScore << endl;
	//delete []tfeaProbalisitic;
	//delete[] tfealabels;
	//delete []labelsIndex;
	return AreaScore;
}

double AdaBoostClassifier::Trap_Area(double x1,double x2,double y1,double y2)
{
	double base = abs(x1-x2);
	double height = (double)(y1+y2)/2.0;
	return base*height;
}

void AdaBoostClassifier::sampleTrainSet(int * data,double * weights,int n,int nSampled,ofstream& fileused)
{
	int* indexTrain = new int[n];
	double* tweights = new double[n];
	for(int i = 0; i< n; i++){
		indexTrain[i] = i;
	//	trainlogfile << data[i] << ' ' << flush;
		tweights[i] = weights[i];
	}
	//trainlogfile << endl;
	QuickSortDescend(tweights,indexTrain,n);
	//for (int k = 0; k < n; k++){
	//	trainlogfile << tweights[k] << ' ' << flush;
	//	if((k+1)%100 == 0) trainlogfile <<endl;
	//}
	//for (int k = 0; k < n; k++){
	//	trainlogfile << indexTrain[k] << ' ' << flush;
	//	if((k+1)%100 == 0) trainlogfile <<endl;
	//}
	for(int i = 0; i < nSampled; i++){
		trainSet_sampled.push_back(data[indexTrain[i]]);
		fileused << data[indexTrain[i]]  << " ";
	}
	delete[] indexTrain;indexTrain=NULL;
	delete []tweights;tweights=NULL;
}
void AdaBoostClassifier::sampleTrainFea(vector<Feature>& xfea,int nSampled)
{
	for(int i = 0; i < totalfeatures; i++)
		for(int j = 0; j < nSampled*2; j++){
			xfea.push_back(FeatureSet[i*(totalcount)+trainSet_sampled[j]]);
		//	trainlogfile << "image# " <<  trainSet_sampled[j] << "'s features: ";
	/*		for(int k =0; k < 32;k++){
				trainlogfile << xfea[j].fea[k] << ' ' << flush;
			}
			trainlogfile << endl;*/
		}
}

