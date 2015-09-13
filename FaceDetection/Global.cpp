#include "stdafx.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cstdlib>
#include <cv.h>
#include <highgui.h>
#include <io.h>

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

typedef struct Weight{
	double wei[33];
} Weight;

CString option_filename = _T("info/options.txt");
CString train_log_filename = _T("info/trainlog.txt");
string cascade_filename;
string positive_filename;
string negative_dirname;
string boosting_positive;
string boosting_negative;

ofstream rocfile;
ifstream weightsfile;
ofstream trainlogfile;
int sx; //width
int sy; //height
int max_nodes;
double gFPR;// over all FPR
double gHitrate;// minimum hit-rate per stage
double galpha; // alpha in Logistic
double gdelta; // delta in Logistic
BOOL   gBoostedp; // in Cascade,is positive Boosted?
int gTotalX_Plus;//number of train files for each cascade
int gsampled;// number of train files for positive(negative) for each turn in adaboost
int gCombine_min; // combine_min in Cascade::Post()
vector<string> Boosting_positive_file;
vector<string> Boosting_negative_file;
vector<int> nof;
CascadeClassifier* cascade=NULL;


Feature* FeatureSet=NULL;
int totalfeatures;
int facecount,nonfacecount;
int totalcount,negative_validcount,positive_validcount;
FDImage* trainset=NULL;
int* trainset_Cur = NULL;
BOOL* labels=NULL;

typedef struct patch{
	int x,y,rh,rw;
}patch;

vector<patch> patches;
////////////////////////////////////////////////	Init
void IgnoreComments(ifstream& f)
{
	f.ignore(65536,'\n');
}
void ReadlnString(ifstream& f, string& s)
{
	char buf[256];
	f.getline(buf,255,'\n');
	s = buf;
}
void LoadOptions()
{
	ifstream f;
	int i;

	f.open(option_filename);
	if(f == NULL){
		AfxMessageBox("No options file");
		exit(0);
	}
	IgnoreComments(f); ReadlnString(f,cascade_filename);
	IgnoreComments(f); ReadlnString(f,positive_filename);
	IgnoreComments(f); ReadlnString(f,negative_dirname);
	IgnoreComments(f); ReadlnString(f,boosting_positive);
	IgnoreComments(f); ReadlnString(f,boosting_negative);
	
	IgnoreComments(f); f>>gCombine_min; IgnoreComments(f);
	IgnoreComments(f); f>>gBoostedp; IgnoreComments(f);
	IgnoreComments(f); f>>gdelta; IgnoreComments(f);
	IgnoreComments(f); f>>galpha; IgnoreComments(f);
	IgnoreComments(f); f>>sx; IgnoreComments(f);
	IgnoreComments(f); f>>sy; IgnoreComments(f);
	IgnoreComments(f); f>>max_nodes; IgnoreComments(f);

	IgnoreComments(f); f>>gFPR; IgnoreComments(f);
	IgnoreComments(f); f>>gHitrate; IgnoreComments(f);
	IgnoreComments(f); f>>gTotalX_Plus; IgnoreComments(f);
	IgnoreComments(f); f>>gsampled; IgnoreComments(f);

	nof.resize(max_nodes);
	IgnoreComments(f); for(i=0;i<max_nodes;i++) f>>nof[i]; 	IgnoreComments(f);
	f.close();
}
void InitGlobalData()
{
	srand((unsigned)time(NULL));
	LoadOptions();
	cascade = new CascadeClassifier;
	ASSERT(cascade != NULL);


	//fdefeau.open("data/ada1.txt", ios_base::binary | ios_base::in);
}
void ClearUpGlobalData()
{
	if(FeatureSet != NULL){
		delete []FeatureSet;
		FeatureSet=NULL;
	}
	if(trainset != NULL){
		delete []trainset;
		trainset=NULL;
	}
	if(trainset_Cur != NULL){
		delete []trainset_Cur;
		trainset_Cur=NULL;
	}
	if(labels != NULL){
		delete []labels;
		labels=NULL;
	}
	delete cascade; cascade = NULL;

}
////////////////////////////////////////////////	Patch
void checkpatches()
{
	int n = (int)patches.size();
	ofstream f("info/checkpatches.txt");

	for(int i = 0 ; i < n; i++){
		if(((patches[i].x + patches[i].rw-1) > sx )|| ((patches[i].y + patches[i].rh-1) > sy))
			f << patches[i].x << ' ' << patches[i].y << ' ' << patches[i].rw << ' ' << patches[i].rh << endl;
	}
	f.close();
}
void GeneratePatches()
{
	int x,y;
	patch tpatch;
	ofstream f("info/patches.txt");
	int step = 4;
	int rw ,rh;
	// patches of 2x2,allow 1:1,1:2,2:1,2:3,3:2 aspect ratio
	f << "1 : 1 " << endl;
	for(rh = 12; rh <= sy ; rh += 8){
		//1:1
		rw = rh;
		for(y = 0; y <= sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}
	f << "1 : 2 " << endl;
	for(rw = 24; rw <= sx ; rw += 16){
		//1:1
		rh = rw/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "2 : 1 " << endl;
	for(rh = 24; rh <= sy ; rh += 16){
		//2:1
		rw = rh/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "2 : 3 " << endl;
	for(rh = 12; rh <= sy ; rh += 8 ){
		//2:3
		rw = rh * 3/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "3 : 2 " << endl;
	for(rh = 18; rh <= sy ; rh += 12){
		//3:2
		rw = rh* 2/3;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	} 
	f << "4 : 1 " << endl;
	// patches of 4x1,allow 4:1 aspect ratio
	for(rh = 24; rh <= sy ; rh += 16 ){
		rw = rh/4;
		for(y = 0; y <= sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}

	f << "1 : 4 " << endl;
	// patches of 1x4,allow 1:4 aspect ratio
	for(rh = 6; rh <= sy ; rh += 16){
		rw = rh*4;
		for(y = 0; y < sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}
	f.close();
	totalfeatures = (int)patches.size();
	checkpatches();
}
void GeneratePatches2()
{
	int x,y;
	patch tpatch;
	ofstream f("info/patches.txt");
	int step = 4;
	int rw ,rh;
	// patches of 2x2,allow 1:1,1:2,2:1,2:3,3:2 aspect ratio
	f << "1 : 1 " << endl;
	for(rh = 12; rh <= sy ; rh += 8){
		//1:1
		rw = rh;
		for(y = 0; y <= sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}
	f << "1 : 2 " << endl;
	for(rw = 24; rw <= sx ; rw += 8){
		//1:2
		rh = rw/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "2 : 1 " << endl;
	for(rh = 24; rh <= sy ; rh += 8){
		//2:1
		rw = rh/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "2 : 3 " << endl;
	for(rh = 12; rh <= sy ; rh += 4 ){
		//2:3
		rw = rh * 3/2;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	}
	f << "3 : 2 " << endl;
	for(rh = 18; rh <= sy ; rh += 6){
		//3:2
		rw = rh* 2/3;
		for(y = 0; y <= sy-rh; y += step){
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
		}
	} 
	f << "4 : 1 " << endl;
	// patches of 4x1,allow 4:1 aspect ratio
	for(rh = 24; rh <= sy ; rh += 16 ){
		rw = rh/4;
		for(y = 0; y <= sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}

	f << "1 : 4 " << endl;
	// patches of 1x4,allow 1:4 aspect ratio
	for(rh = 6; rh <= sy ; rh += 4){
		rw = rh*4;
		for(y = 0; y < sy-rh; y += step)
			for(x = 0; x <= sx-rw; x += step){
				tpatch.x = x;tpatch.y = y;tpatch.rw	= rw;tpatch.rh = rh;
				patches.push_back(tpatch);
				f << x << ' ' << y << ' ' << rw << ' ' << rh << endl;
			}
	}
	f.close();
	totalfeatures = (int)patches.size();
	checkpatches();
}
////////////////////////////////////////////////	Train Set
int dir_file_list(string folderPath, vector<string> &file_vec) {
	_finddata_t FileInfo;
	string strfind = folderPath + "\\*";
	long Handle = (long)_findfirst(strfind.c_str(), &FileInfo);

	if (Handle == -1L){
		return 0;
	}
	do{
		//判断是否有子目录
		if (!(FileInfo.attrib & _A_SUBDIR)){
			file_vec.push_back(string(folderPath+"/"+FileInfo.name));
		}
	}while (_findnext(Handle, &FileInfo) == 0);

	_findclose(Handle);
	return 1;
}
void ReadOneTrainingSample(ifstream& is,FDImage& image)
{
	int i,j;
	char buf[256];

	FDImage timage;
	ASSERT(sx<=256 && sy<=256);
	is>>image.label; is.ignore(256,'\n');
	ASSERT( (image.label == 0) || (image.label == 1) );

	is>>timage.height>>timage.width; is.ignore(256,'\n');
	ASSERT(timage.height==sx); 
	ASSERT(timage.width==sy);

	timage.SetSize(CSize(timage.height,timage.width));

	for(i=0;i<timage.height;i++)
	{
		is.read(buf,timage.width);
		for(j=0;j<timage.width;j++) 
		{
			timage.data[i][j] = REAL(int(unsigned char(buf[j-1])));
			ASSERT(timage.data[i][j]>=0 && timage.data[i][j] <= 255);
		}
	}
	is.ignore(256,'\n');
	timage.Resize(image,40.0/24.0);
	timage.Clear();
}
void ReadOneTrainingSamplefromfile(const string& filename,FDImage& image)
{ 

	IplImage* img;

	img =  cvLoadImage(filename.c_str(),0);
	image.SetSize(CSize(img->height,img->width));
	for(int i=0,ih=img->height,iw=img->width;i<ih;i++)
	{
		REAL* pdata = image.data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
		for(int j=0;j<iw;j++) pdata[j] = pimg[j];
	}
	cvReleaseImage(&img);
	img = NULL;
}
int ReadTrainSet()
{
	ifstream f1;
	int i;
	vector<string> file_nonface;
	vector<string> file_face;

	int t = dir_file_list(positive_filename,file_face);
	if(t != 1)	return 0;

	t = dir_file_list(negative_dirname,file_nonface);
	if(t != 1)	return 0;

	t = dir_file_list(boosting_positive,Boosting_positive_file);
	if(t != 1)	return 0;
	else positive_validcount = Boosting_positive_file.size();

	t = dir_file_list(boosting_negative,Boosting_negative_file);
	if(t != 1)	return 0;
	else negative_validcount = Boosting_negative_file.size();

	facecount = gTotalX_Plus;nonfacecount = gTotalX_Plus;
	totalcount  = facecount+nonfacecount;

	if(facecount > (int)file_face.size() || nonfacecount > (int)file_nonface.size())	return 1;

	delete[] trainset; trainset=NULL;delete[] labels;labels = NULL;
	trainset = new FDImage[totalcount]; ASSERT(trainset != NULL);
	labels = new int[totalcount]; ASSERT(labels != NULL);

	for(i = 0; i < facecount; i++){
		ReadOneTrainingSamplefromfile(file_face[i],trainset[i]);
		labels[i] = 1;trainset[i].label=1;
	}
	for(i= facecount;i< facecount+nonfacecount; i++) {
		ReadOneTrainingSamplefromfile(file_nonface[i-facecount],trainset[i]);
		labels[i] = -1;trainset[i].label=0;
	}

	return 2;
}
////////////////////////////////////////////////	Sort
//void QuickSortAscend(double* values,int* vindex,const int n)
//{
//	int i,j;
//	double tmp;
//	int index = 0;
//	//for(int k = 0; k < n; k++){
//	//	trainlogfile <<" p:" << values[k] << ' ' << "Lindex:" << vindex[k] << ";  " << endl;
//	//}
//
//	for(i=0;i < n; i++){
//		tmp = values[i];
//		index = i;
//		for( j =i+1; j < n;j++ ){
//			if(values[j] < tmp){
//				tmp = values[j];
//				index = j;
//			}
//		}
//		if(index != i){
//			values[index] = values[i];
//			values[i] = tmp;
//			vindex[index] = vindex[i];
//			vindex[i] = index;
//		}
//
//	}
//
//
//	return;
//}
void QuickSortDescend(double* values,int* vindex,const int n)
{
	int i,j;
	double tmp;
	int index = 0,tmpIndex;
	for(i=0;i < n; i++){
		tmp = values[i];
		index = i;
		for( j =i+1; j < n;j++ ){
			if(values[j] > tmp){
				tmp = values[j];
				index = j;
			}
		}
		if(index != i){
			values[index] = values[i];
			values[i] = tmp;
			tmpIndex = vindex[index];
			vindex[index] = vindex[i];
			vindex[i] = tmpIndex;
		}
	}

	return;
}
////////////////////////////////////////////////   Feature
void FeatureNormalization(double *fea,double d){
	double theta = (double)2/sqrt(d);
	double e = 0.01;
	double *u = new double[d];
	double sqsum = 0.0;
	for(int k=0; k < d;k++) sqsum += fea[k]*fea[k];
	//trainlogfile<<"sum of fea^2: " << sqsum;	
	sqsum += e;sqsum = sqrt(sqsum);
	//trainlogfile<<"  sqsum of fea^2: " << sqsum<<endl;	
	for(int k=0; k < d;k++) u[k] = fea[k]/sqsum;
	for(int k=0; k < d;k++){
		if(u[k] > theta)
			u[k] = theta;
		else if(u[k] < -theta)
			u[k] = -theta;
	}
	sqsum = 0.0;
	for(int k=0; k < d;k++) sqsum += u[k]*u[k];
	//trainlogfile<<"sum of u^2 " << sqsum;	
	sqsum += e;sqsum = sqrt(sqsum);
	//trainlogfile<<"sqsum of u^2 " << sqsum<<endl;	
	for(int k=0; k < d;k++) fea[k] = u[k]/sqsum;
	delete []u;u=NULL;
}
void getFeatureDescriptor(FDImage &im,double* pFea,int x,int y,int rh,int rw)
{
	int pointIndex[4];
	//x++;y++;
//	trainlogfile << x << " " << y << " " << rh << " "<< rw << endl;
	int width = im.width+1;int height = im.height+1;
	if(rh == (rw*4)){
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
		pointIndex[0] = y*width+x;pointIndex[1] = pointIndex[0]+rw;
		pointIndex[2] =pointIndex[0]+rh/4*width;pointIndex[3] = pointIndex[2]+rw;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rh/4*width;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else if(rh == (rw/4)){
		// 1x4 cells
		// |------|------|------|------|
		// |   1  |	 2   |	 3  |	4  |
		// |------|------|------|------|
		pointIndex[0] = y*width+x;pointIndex[1] = pointIndex[0]+rw/4;
		pointIndex[2] = pointIndex[0]+rh*width;pointIndex[3] = pointIndex[2]+rw/4;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rw/4;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else{
		// 2x2 cells
		// |------|------|
		// |   1  |  2   |
		// |------|------|
		// |   3  |  4   |
		// |------|------|
		pointIndex[0] = y*width+x;pointIndex[1] =pointIndex[0]+rw/2;
		pointIndex[2] = pointIndex[0]+rh/2*width;pointIndex[3] = pointIndex[2]+rw/2;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
	//	trainlogfile << " fea[0] " << pFea[0]<< endl;
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+8,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
	//	trainlogfile << " fea[0] " << pFea[8]<< endl;
		pointIndex[0] -= rw/2;pointIndex[1] -= rw/2;pointIndex[2] -= rw/2;pointIndex[3] -= rw/2;
		pointIndex[0] += rh/2*width;pointIndex[1] += rh/2*width;pointIndex[2] += rh/2*width;pointIndex[3] += rh/2*width;
		getRectFeatureDescriptor(im,pFea+16,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
	//	trainlogfile <<  " fea[0] " << pFea[16] << endl;
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+24,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
	//	trainlogfile << " fea[0] " <<  pFea[24] << endl;
		
	}
	FeatureNormalization(pFea,32);
}
void getFeatureDescriptor(FDImage &im,double* pFea,int x,int y,int rh,int rw,double ratio)
{
	int pointIndex[4];
	int width = im.width+1;int height = im.height+1;
	if(ratio == 4){
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
		pointIndex[0] = y*width+x;pointIndex[1] = pointIndex[0]+rw;
		pointIndex[2] =pointIndex[0]+rh/4*width;pointIndex[3] = pointIndex[2]+rw;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rh/4*width;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else if(ratio == 0.25){
		// 1x4 cells
		// |------|------|------|------|
		// |   1  |	 2   |	 3  |	4  |
		// |------|------|------|------|
		pointIndex[0] = y*width+x;pointIndex[1] = pointIndex[0]+rw/4;
		pointIndex[2] = pointIndex[0]+rh*width;pointIndex[3] = pointIndex[2]+rw/4;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		for(int j = 0; j < 3; j++){
			for(int i=0;i < 4;i++){
				pointIndex[i] += rw/4;}
			getRectFeatureDescriptor(im,pFea+8*(j+1),pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		}
	}else{
		// 2x2 cells
		// |------|------|
		// |   1  |  2   |
		// |------|------|
		// |   3  |  4   |
		// |------|------|
		pointIndex[0] = y*width+x;pointIndex[1] =pointIndex[0]+rw/2;
		pointIndex[2] = pointIndex[0]+rh/2*width;pointIndex[3] = pointIndex[2]+rw/2;
		getRectFeatureDescriptor(im,pFea,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		//	trainlogfile << " fea[0] " << pFea[0]<< endl;
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+8,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		//	trainlogfile << " fea[0] " << pFea[8]<< endl;
		pointIndex[0] -= rw/2;pointIndex[1] -= rw/2;pointIndex[2] -= rw/2;pointIndex[3] -= rw/2;
		pointIndex[0] += rh/2*width;pointIndex[1] += rh/2*width;pointIndex[2] += rh/2*width;pointIndex[3] += rh/2*width;
		getRectFeatureDescriptor(im,pFea+16,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		//	trainlogfile <<  " fea[0] " << pFea[16] << endl;
		pointIndex[0] += rw/2;pointIndex[1] += rw/2;pointIndex[2] += rw/2;pointIndex[3] += rw/2;
		getRectFeatureDescriptor(im,pFea+24,pointIndex[0],pointIndex[1],pointIndex[2],pointIndex[3]);
		//	trainlogfile << " fea[0] " <<  pFea[24] << endl;

	}
	FeatureNormalization(pFea,32);
}
void getRectFeatureDescriptor(FDImage &im,double* pFea,int p1,int p2,int p3,int p4)
{
	double* a1 = (im.m_integralIamge[p1].f);
	double* a2 = (im.m_integralIamge[p2].f);
	double* a3 = (im.m_integralIamge[p3].f);
	double* a4 = (im.m_integralIamge[p4].f);

	pFea[0] = (a4[0] - a2[0] - a3[0] + a1[0]);
	//trainlogfile << p4 << ' ' << p3 << ' ' << p2 << ' ' << p1 <<endl;
//	trainlogfile << a4[0] << ' ' << a3[0] << ' ' << a2[0] << ' ' << a1[0] <<endl;
	pFea[1] = (a4[1] - a2[1] - a3[1] + a1[1]);
	pFea[2] = (a4[2] - a2[2] - a3[2] + a1[2]);
	pFea[3] = (a4[3] - a2[3] - a3[3] + a1[3]);
	pFea[4] = (a4[4] - a2[4] - a3[4] + a1[4]);
	pFea[5] = (a4[5] - a2[5] - a3[5] + a1[5]);
	pFea[6] = (a4[6] - a2[6] - a3[6] + a1[6]);
	pFea[7] = (a4[7] - a2[7] - a3[7] + a1[7]);

}
void GetFeatureSet()
{
//	ofstream fFea;
//	fFea.open("data/fea.txt",ios_base::binary | ios_base::out);
	if(FeatureSet != NULL)	delete[] FeatureSet;

	FeatureSet = new Feature[totalfeatures*totalcount];
	for(int i = 0; i < totalfeatures; i++){
	//	fFea << "patch: " << i << endl;
		for(int j = 0; j < (totalcount); j++){
	//		trainlogfile << "pathes" << i <<  "on image " << j << "features" <<endl;
			getFeatureDescriptor(trainset[j],FeatureSet[i*(totalcount)+j].fea,patches[i].x,patches[i].y,patches[i].rh,patches[i].rw);
		//	for (int k =0; k < 32; k++) fFea  << FeatureSet[i*totalfeatures+j].fea[k] << ' ';trainlogfile<<endl;	
		}
	}
}
////////////////////////////////////////////////   Train
double sigmoid(double x)
{
	double ans = exp(x) / (1 + exp(x));  
	return ans; 
}
double WX(const vector<double>& w,const Feature& data)  
{  
	double x = 0.0;
	x  += w[0];
	for(int k =1;k<33;k++)
		x += w[k]*data.fea[k-1];
	return x;
}  
double sigmoid(const vector<double>& w,const Feature& data)  
{  
	double x = WX(w,data);
	double ans = exp(x) / (1 + exp(x));  
	return ans;  
} 
double CostFunction(vector<double> &w,vector<Feature> &data,vector<int> &y,int ifea,int totalx)  
{  
	//double ans = 0;  
	//for(int i=0;i<1280;i++)  
	//{  
	//	double x = WX(w,data[ifea*1280+i]);  
	//	ans += y[i] * x - log(1 + exp(x));  
	//}  
	//return ans;  

	// Machine learning,Andrew NG-2012
	double ans = 0; // double lam = 10000000;double wSum = 0.0;
	double sigX ;
	for(int i=0;i<totalx;i++)  
	{  
		sigX = sigmoid(w,data[ifea*totalx+i]);
		ans += y[i] * log(sigX) + (1-y[i])*log(1-sigX);  
	}  
//	for(int i = 0; i< w.size();i++)	wSum += w[i]*w[i];
//	wSum = lam /(2*1280)*wSum;
	ans	/= totalx;
//	ans = -ans + wSum;
	return ans;  
}  
void gradient(double alpha,vector<double> &w,vector<Feature> &data,vector<int> &y,int ifea,int Index,int totalx)  
{  
	
	double tmp ; 
	tmp = 0;
	tmp +=  (sigmoid(w,data[ifea*totalx+Index])- y[Index] );  
	w[0] -=alpha * tmp;  
	for(UINT i=1;i<w.size();i++) {  
		tmp = 0;   
		tmp +=  data[ifea*totalx+Index].fea[i-1] * (sigmoid(w,data[ifea*totalx+Index])- y[Index] );  
		w[i] -= alpha *tmp;  
	}  

/*
	for(int j=0;j<1280;j++)  
		tmp +=  (y[j] - sigmoid(w,data[ifea*1280+j]));  
	w[0] +=alpha * tmp;  

	for(int i=1;i<w.size();i++)  
	{  
		tmp = 0;  
		for(int j=0;j<1280;j++)  
			tmp += alpha * data[ifea*1280+j].fea[i-1] * (y[j] - sigmoid(w,data[ifea*1280+j]));  
		w[i] += tmp;  
	}
	*/
}  
void Logistic(vector<Feature> &x,vector<int> &y,vector<double> &w,int ifea,int totalx)  
{  
	int cnt = 0;  
	vector<int> dataIndex;
	double alpha;
	int randIndex;
	double delta = gdelta;  
	double oldCost = CostFunction(w,x,y,ifea,totalx);
	alpha = 4/(0+0+1.0) + galpha;
	gradient(alpha,w,x,y,ifea,0,totalx); 
	double newCost = CostFunction(w,x,y,ifea,totalx);
	srand((int)time(0)); 
	double* h = new double[totalx];
	//trainlogfile << oldCost  << " " << newCost << ";";
	while(fabs(newCost - oldCost) > delta) {  
	//	trainlogfile << oldCost  << " " << newCost << ";";
		cnt++;

		oldCost = newCost;
		randIndex = 0;

		for(int j = 0; j < totalx;j++)
			dataIndex.push_back(j);

		for(int k = 0; k < totalx;k++){
			alpha = 4/(cnt+k+1.0) + galpha;
			randIndex = rand()%dataIndex.size();
			gradient(alpha,w,x,y,ifea,dataIndex[randIndex],totalx);
			dataIndex.erase(dataIndex.begin()+randIndex);
			if(dataIndex.size() > 0) randIndex = rand()%dataIndex.size();
		}	
		//for(int j = 0; j < 1280; j++){
		//	double tmp = w[0] * 1;
		//	for(int k = 0; k < 32; k++){
		//		tmp += x[j].fea[k]*w[k+1];
		//	}
		//	h[j] = sigmoid(tmp);
		//}
		//// x0 = 1
		//for(int j = 0;j< 1; j++){
		//	double tmp = 0;  
		//	for(int k=0;k<1280;k++)
		//		tmp += (h[k]-y[k]);  
		//	w[j] -= alpha*tmp;
		//	w[j] = w[j] + alpha*10000000/1280*w[j];
		//}
		//for(int j = 1;j< 33; j++){
		//	double tmp = 0;  
		//	for(int k=0;k<1280;k++)
		//		tmp += x[k].fea[j-1] * (h[k]-y[k]);  
		//	w[j] -=  alpha *tmp;
		//	w[j] = w[j] + alpha*10000000/1280*w[j];
		//}

	/*	for(int i = 0; i< w.size();i++){
			w[i] = w[i] + alpha*10000/1280*w[i];
		}*/
		newCost = CostFunction(w,x,y,ifea,totalx);  
		dataIndex.clear();
		/*			trainlogfile << "counts#:" << cnt << endl;
		trainlogfile << "y[0]:" << y[0] << ";" << "h[0]:" << sigmoid(w,x[0])<< "y[0] - h[0]"<< y[0]-sigmoid(w,x[0]) << endl;
		for(int j = 0; j < 33; j++){
		trainlogfile << w[j] << ' ' <<flush;
		}
		trainlogfile <<endl << fabs(newLw - objLw) << endl;
		}  */
		//trainlogfile<< "cnt :" << cnt;
		//trainlogfile << ";y[0]:" << y[0] << ";" << "h[0]:" << sigmoid(w,x[0])<< "y[0] - h[0]"<< y[0]-sigmoid(w,x[0]) << endl;
	}
} 
void trainLogistic(vector<Feature> &xfea,vector<int>& ylabel,vector<WeakClassifier>& hk,int totalx) 
{  
	char tmpbuf[128];

	/* Set time zone from TZ environment variable. If TZ is not set,
     * the operating system is queried to obtain the default value 
     * for the variable. 
     */
    _tzset();

 
    _strtime( tmpbuf );
	trainlogfile << "start time : "<< tmpbuf << " " << totalx << endl;

	WeakClassifier th;
	vector<double> weights;
	for(int i =0; i < totalfeatures; i++){	
		weights.clear();
		for(int k=0;k<33;k++) weights.push_back(0);  
		Logistic(xfea,ylabel,weights,i,totalx);
		for(int j = 0; j < 33; j++){
			th.weights[j] = weights[j];
		}
		FeatureNormalization(th.weights,33);
		th.x = patches[i].x;th.y = patches[i].y;th.rh = patches[i].rh;th.rw = patches[i].rw;
		hk.push_back(th);
	//	weightsfile << "hk[" <<  i << "]: " << "weights: ";
//		for(int k = 0; k < 33;k++) weightsfile << th.weights[k] << ' ';
		//weightsfile << endl;
		
	}
//	weightsfile << "one round ends: h[]'s size: " << hk.size()<< endl; 
	_strtime( tmpbuf );
	trainlogfile << "end time : "<< tmpbuf << endl;
}  
