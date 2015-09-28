#include "stdafx.h"
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <algorithm>
#include <mmsystem.h>
#include <set>
#include <shlobj.h>
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

CascadeClassifier::CascadeClassifier():count(0),indexBoosted(0)
{
}

CascadeClassifier::~CascadeClassifier()
{
	Clear();	
}

void CascadeClassifier::Clear()
{
	count = 0;
	strongClassifiers.clear();
}

CascadeClassifier& CascadeClassifier::operator=(const CascadeClassifier& source)
{
	Clear();
	count = source.count;
	strongClassifiers = source.strongClassifiers;
	return *this;
}

void CascadeClassifier::ReadFromFile(ifstream& f)
{
	Clear();
	f>>count; f.ignore(256,'\n');
	AdaBoostClassifier tstrongclassifier;
	strongClassifiers.clear();
	for(int i=0;i<count;i++){
		tstrongclassifier.ReadFromFile(f);
		strongClassifiers.push_back(tstrongclassifier);
	}
}

void CascadeClassifier::WriteToFile(ofstream& f)
{
	count = (int)strongClassifiers.size();
	f<<count<<endl;
	for(int i=0;i<count;i++) 
		strongClassifiers[i].WriteToFile(f);
}

void CascadeClassifier::LoadDefaultCascade()
{
	ifstream f;
	f.open("data/cascadeClassifier.txt");
	if(f == NULL){
		AfxMessageBox("No classifier");
		return ;
	}

	ReadFromFile(f);
	f.close();
}

void CascadeClassifier::DrawResults(FDImage& image,const vector<CRect>& results) const
{
	int i;
	unsigned int k;
	int x1,x2,y1,y2;

	for(k=0;k<results.size();k++)
	{
		y1 = (results[k].top>=0)?results[k].top:0; 
		y1 = (results[k].top<image.height)?results[k].top:(image.height-1);
		y2 = (results[k].bottom>=0)?results[k].bottom:0;
		y2 = (results[k].bottom<image.height)?results[k].bottom:(image.height-1);
		x1 = (results[k].left>=0)?results[k].left:0;
		x1 = (results[k].left<image.width)?results[k].left:(image.width-1);
		x2 = (results[k].right>=0)?results[k].right:0;
		x2 = (results[k].right<image.width)?results[k].right:(image.width-1);	
		for(i=y1;i<=y2;i++) 
		{
			image.data[i][x1] = 255;
			image.data[i][x2] = 255;
		}
		for(i=x1;i<=x2;i++)
		{
			image.data[y1][i] = 255;
			image.data[y2][i] = 255;
		}
	}
}

double CascadeClassifier::Apply(FDImage &im)
{
	for(int i = 0; i < count ; i++){
		if(strongClassifiers[i].Apply(im) == 0)
			return 0;
	}
	return 1;
}

double CascadeClassifier::ApplyValue(FDImage &im,double dif)
{
	for(int i = 0; i < count ; i++){
		if( (strongClassifiers[i].GetValue(im) -strongClassifiers[i].thresh) < -dif)
			return 0;
	}
	return 1;
}

void CascadeClassifier::FaceDetect_ScaleImage(FDImage& original/*,const CString filename*/) 
{
	FDImage procface;
	FDImage image,square;
	double value;
	int result;
	CRect rect;
	REAL ratio;
	vector<CRect> results;
//	ofstream detectionfile;
	//detectionfile.open("data/dete.txt");
	//procface.Copy(original);

	ratio = 1.2;
	original.Resize(procface,ratio);
	results.clear();
//	REAL paddedsize = REAL(1)/REAL((sx+1)*(sy+1));
//	detectionfile << "image size" << procface.height << " " <<procface.width << endl;
	while((procface.height+1>sx+1) && (procface.width+1>sy+1))
	{
		//	detectionfile << "sub window size" << (sx+1)*ratio << " " << (sy+1)*ratio<<endl;
		//detectionfile << "sub window pos:" << endl;
	//	procface.CalcSquareAndIntegral(square,image);
		procface.CalcgradientImage();
		procface.calcintegralImage();
		procface.cleartmp();
		for(int i=0,size_x=procface.height+1-sx;i<size_x;i+=2)
			for(int j=0,size_y=procface.width+1-sy;j<size_y;j+=2){

				result = 1;
				for(int k=0;k<count;k++)
				{
					value = 0.0;
					//detectionfile << "stage: " << k << endl;
					for(int t=0,size=strongClassifiers[k].count;t<size;t++)
					{
						//	REAL f1 = 0;
						//	REAL** p = image.data + i;
						double yfea=0.0;
						WeakClassifier& s = strongClassifiers[k].weakClassifiers[t];;
						double tfea[32];
						//detectionfile << "patch size and pos" << s.x+j << " " << s.y+i << " " << s.rh*ratio<<" " << s.rw*ratio<<endl;
						getFeatureDescriptor(procface,tfea,s.x+j,s.y+i,s.rh,s.rw);

						yfea += s.weights[0];
						//detectionfile << "tfea: " ; 
						for(int u =0; u < 32;u++){
							yfea += tfea[u]*s.weights[u+1];
						//	detectionfile << yfea << " " ;
						}
					//	detectionfile << endl;
						value += sigmoid(yfea);
					//	detectionfile << "classifier: " << t <<  "yfea: " << yfea << "value: " << value <<endl;   
					}
					value /=(double)strongClassifiers[k].count;
				//	detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
					if(value <= strongClassifiers[k].thresh){
						result = 0;
						break;
					}
				}
				if(result!=0) 
				{
					const REAL r = 1.0/ratio;
					rect.left = (LONG)(j*r);rect.top = (LONG)(i*r);
					rect.right = (LONG)((j+sx)*r);rect.bottom = (LONG)((i+sy)*r);
					results.push_back(rect);
					//	detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
				}
			}
		ratio = ratio*0.8;
		procface.Clear();
	///	image.Clear();square.Clear();
		original.Resize(procface,ratio);
	}

	//total_fp += results.size();

	PostProcess(results,gCombine_min);
	PostProcess(results,0);
	DrawResults(original,results);
	//original.Save(filename+"_result.JPG");
}

void CascadeClassifier::FaceDetect_ScaleTempl(FDImage& original/*,const CString filename*/) 
{
	FDImage procface;
	int result;
	CRect rect;
	REAL ratio;
	vector<CRect> results;
	ofstream detectionfile;
	//detectionfile.open("data/dete.txt");
	procface.Copy(original);
	//original.Resize(procface,2.0);
	procface.CalcgradientImage();
	procface.calcintegralImage();
	procface.cleartmp();
	ratio = 1;
	results.clear();
	//detectionfile << "image size" << procface.height << " " <<procface.width << endl;
	while((procface.height>(sx)*ratio) && (procface.width>(sy)*ratio))
	{
		//detectionfile << "sub window size" << (sx+1)*ratio << " " << (sy+1)*ratio<<endl;
		//	detectionfile << "sub window pos:" << endl;
		int nstep = (int)(ratio*sx)/20;
		for(int i=0,size_x=(int)procface.height-(sx*ratio);i<size_x;i+=nstep)
			for(int j=0,size_y=(int)procface.width-(sy*ratio);j<size_y;j+=nstep){
				///detectionfile << "(" << i << "," << j <<")" <<endl;	
				result = 1;
				for(int k=0;k<count;k++)
				{
					double value = 0.0;
					//	detectionfile << "stage: " << k << endl;
					for(int t=0,size=strongClassifiers[k].count;t<size;t++)
					{
						//	REAL f1 = 0;
						//	REAL** p = image.data + i;
						double yfea=0.0;
						WeakClassifier& s = strongClassifiers[k].weakClassifiers[t];;
						double tfea[32];
						//	detectionfile << "patch size and pos" << s.x+j << " " << s.y+i << " " << s.rh*ratio<<" " << s.rw*ratio<< ' '<< (double)s.rh/s.rw << endl;
						getFeatureDescriptor(procface,tfea,j+s.x*ratio,i+s.y*ratio,s.rh*ratio,s.rw*ratio/*,(double)s.rh/s.rw*/);

						yfea += s.weights[0];
						for(int u =0; u < 32;u++)
							yfea += tfea[u]*s.weights[u+1];
						value += sigmoid(yfea);
						//		detectionfile << "classifier: " << t <<  "yfea: " << yfea << "value: " << value <<endl;   
					}
					value /= (double)strongClassifiers[k].count;
					//	detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
					if(value <= strongClassifiers[k].thresh){
						result = 0;
						break;
					}
				}
				if(result!=0) 
				{
					rect.left = (LONG)(j);rect.top = (LONG)(i);
					rect.right = (LONG)(j+sx*ratio);rect.bottom = (LONG)(i+sy*ratio);
					results.push_back(rect);
					//detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
				}
			}
		ratio = ratio*1.2;
	}

	//total_fp += results.size();

	PostProcess(results,gCombine_min);
	PostProcess(results,0);
	DrawResults(original,results);
	//original.Save(filename+"_result.JPG");
}

int CascadeClassifier::FaceDetectWithRet(FDImage& original,const string filename) 
{
	FDImage procface;
	int result;
	CRect rect;
	REAL ratio;
	vector<CRect> results;
	ofstream detectionfile;
	//detectionfile.open("data/dete.txt");
	procface.Copy(original);
	//original.Resize(procface,2.0);
	procface.CalcgradientImage();
	procface.calcintegralImage();
	procface.cleartmp();
	ratio = 1;
	results.clear();
	//detectionfile << "image size" << procface.height << " " <<procface.width << endl;
	while((procface.height>(sx)*ratio) && (procface.width>(sy)*ratio))
	{
		//detectionfile << "sub window size" << (sx+1)*ratio << " " << (sy+1)*ratio<<endl;
		//	detectionfile << "sub window pos:" << endl;
		int nstep = (int)(ratio*sx)/20;
		for(int i=0,size_x=(int)procface.height-(sx*ratio);i<size_x;i+=nstep)
			for(int j=0,size_y=(int)procface.width-(sy*ratio);j<size_y;j+=nstep){
				///detectionfile << "(" << i << "," << j <<")" <<endl;	
				result = 1;
				for(int k=0;k<count;k++)
				{
					double value = 0.0;
					//	detectionfile << "stage: " << k << endl;
					for(int t=0,size=strongClassifiers[k].count;t<size;t++)
					{
						//	REAL f1 = 0;
						//	REAL** p = image.data + i;
						double yfea=0.0;
						WeakClassifier& s = strongClassifiers[k].weakClassifiers[t];;
						double tfea[32];
						//	detectionfile << "patch size and pos" << s.x+j << " " << s.y+i << " " << s.rh*ratio<<" " << s.rw*ratio<< ' '<< (double)s.rh/s.rw << endl;
						getFeatureDescriptor(procface,tfea,j+s.x*ratio,i+s.y*ratio,s.rh*ratio,s.rw*ratio/*,(double)s.rh/s.rw*/);

						yfea += s.weights[0];
						for(int u =0; u < 32;u++)
							yfea += tfea[u]*s.weights[u+1];
						value += sigmoid(yfea);
						//		detectionfile << "classifier: " << t <<  "yfea: " << yfea << "value: " << value <<endl;   
					}
					value /= (double)strongClassifiers[k].count;
					//	detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
					if(value <= strongClassifiers[k].thresh){
						result = 0;
						break;
					}
				}
				if(result!=0) 
				{
					rect.left = (LONG)(j);rect.top = (LONG)(i);
					rect.right = (LONG)(j+sx*ratio);rect.bottom = (LONG)(i+sy*ratio);
					results.push_back(rect);
					//detectionfile << "value: " << value << "thresh: " << strongClassifiers[k].thresh << endl;
				}
			}
		ratio = ratio*1.2;
	}

	//total_fp += results.size();

	PostProcess(results,gCombine_min);
	PostProcess(results,0);
	DrawResults(original,results);
	original.Save(filename+"_result.JPG");
	return results.size();
}

inline int SizeOfRect(const CRect& rect)
{
	return rect.Height()*rect.Width();
}

void CascadeClassifier::PostProcess(vector<CRect>& result,const int combine_min)
{
	vector<CRect> res1;
	vector<CRect> resmax;
	vector<int> res2;
	bool yet;
	CRect rectInter,rectUnion;

	for(unsigned int i=0,size_i=result.size();i<size_i;i++){
		yet = false;
		CRect& result_i = result[i];
		for(unsigned int j=0,size_r=res1.size();j<size_r;j++){
			CRect& resmax_j = resmax[j];
			if(rectInter.IntersectRect(result_i,resmax_j)&&rectUnion.UnionRect(result_i,resmax_j)){
				if((double)SizeOfRect(rectInter)/SizeOfRect(rectUnion) > 0.2){
					CRect& res1_j = res1[j];
					resmax_j.UnionRect(resmax_j,result_i);
					res1_j.bottom += result_i.bottom;
					res1_j.top += result_i.top;
					res1_j.left += result_i.left;
					res1_j.right += result_i.right;
					res2[j]++;
					yet = true;
					break;
				}
			}
		}
		if(yet==false){
			res1.push_back(result_i);
			resmax.push_back(result_i);
			res2.push_back(1);
		}
	}

	for(unsigned int i=0,size=res1.size();i<size;i++){
		const int count = res2[i];
		CRect& res1_i = res1[i];
		res1_i.top /= count;
		res1_i.bottom /= count;
		res1_i.left /= count;
		res1_i.right /= count;
	}

	vector<CRect> result1,result2;vector<int> res3,res4;
	for(int i=0,size=res1.size();i<size;i++){
		if(res2[i]>combine_min){
			result2.push_back(res1[i]);
			res4.push_back(res2[i]);
		}
	}
	bool bcan;
	while(true){
		bcan = false;
		vector<int> toDel;
		result1.clear();result1 = result2;
		res3.clear();res3 = res4;
		for(int i=0,size_i=result1.size();i<size_i;i++)
			toDel.push_back(1);

		for(int i=0,size_i=result1.size();i<size_i;i++)
		{
			CRect& result_i = result1[i];
			for(unsigned int j=i+1,size_j=result1.size()-1;j<size_j;j++)
			{
				CRect& result_j = result1[j];
				if(rectInter.IntersectRect(result_i,result_j))
				{
					double areaInter = (double)SizeOfRect(rectInter),areaI = (double)SizeOfRect(result_i),areaJ=(double)SizeOfRect(result_j);
					double s1 = (double)(areaInter/areaI)/(areaInter/areaJ);
					double s2 = (double)(areaInter/areaJ)/(areaInter/areaI);
					/*double s1 = areaInter/areaI;
					double s2 = areaInter/areaJ;*/
					if( s1/s2 >  4 || s2/s1 > 4 || areaInter/areaI > 0.6 || areaInter/areaJ > 0.6){
						if(res3[i] > res3[j])
							toDel[j] = 0;
						else
							toDel[i] = 0;
						bcan = true;
					}
				}
			}
		}
		if(bcan == false)
			break;
		result2.clear();res4.clear();
		for(unsigned int i=0,size=result1.size();i<size;i++) 
			if(toDel[i] == 1){
				result2.push_back(result1[i]);
				res4.push_back(res3[i]);
			}
	}
	result.clear();
	result = result2;
}
int CascadeClassifier::BoostingPositiveSet(ofstream &logfile)
{

	double ndiff = 0.0;
	int nSample = 0;int indexBoosted=0;
	FDImage img2scan;
	set<int> used;
	 
	while(nSample < facecount){
		img2scan.Clear();
		ReadOneTrainingSamplefromfile(Boosting_positive_file[indexBoosted],img2scan); 
		img2scan.label = 1;img2scan.CalcgradientImage();img2scan.calcintegralImage();
		if( ((int)ApplyValue(img2scan,ndiff)==0)&& (used.find(indexBoosted)==used.end()) ){
			trainset[nSample].Copy(img2scan);
			trainset[nSample].CalcgradientImage();
			trainset[nSample].calcintegralImage();
			trainset[nSample].cleartmp();
			labels[nSample] = 1;
			used.insert(indexBoosted);
			logfile << indexBoosted << ' ';
		}	

		nSample++;indexBoosted++;
		if( (indexBoosted == positive_validcount) && nSample < facecount){
			indexBoosted = 0;
			ndiff -= 0.0001;
		}
	}

}
int CascadeClassifier::BoostingNegativeSet(ofstream &logfile)
{
	FDImage img2scan,original;
	int result;
	CRect rect;
	vector<CRect> results;
	int nStep = 4;

	int nBoosted=0;
	logfile << endl << "Negative Set:" <<flush; 
	while(nBoosted < facecount && indexBoosted < negative_validcount){

		original.Clear();img2scan.Clear();results.clear();
		ReadOneTrainingSamplefromfile(Boosting_negative_file[indexBoosted],original); 
		logfile << endl << Boosting_negative_file[indexBoosted].c_str() << endl;
		img2scan.Copy(original);img2scan.CalcgradientImage();img2scan.calcintegralImage();img2scan.cleartmp();

		for(int i=0,size_y=img2scan.height-sy;i<size_y;i+=nStep){
			for(int j=0,size_x=img2scan.width-sx;j<size_x;j+=nStep){
				result = 1;
				for(int k=0;k<strongClassifiers.size();k++){
					double value = 0.0;
					for(int t=0,size=strongClassifiers[k].count;t<size;t++){
						double yfea=0.0;
						WeakClassifier& s = strongClassifiers[k].weakClassifiers[t];;
						double tfea[32];
						getFeatureDescriptor(img2scan,tfea,j+s.x,i+s.y,s.rh,s.rw);

						yfea += s.weights[0];
						for(int u =0; u < 32;u++)
							yfea += tfea[u]*s.weights[u+1];
						value += sigmoid(yfea);
					}
					value /= (double)strongClassifiers[k].count;
					if(value <= strongClassifiers[k].thresh){
						result = 0;
						break;
					}
				}
				if(result!=0) {
					rect.left = (LONG)(j);rect.top = (LONG)(i);
					rect.right = (LONG)(j+sx);rect.bottom = (LONG)(i+sy);
					results.push_back(rect);
				}
			}
		}
			
		for(int j = 0; (j < results.size())&& (nBoosted < facecount); j++){
			trainset[nBoosted+facecount].Clear();trainset[nBoosted+facecount].SetSize(CSize(sx,sy));
			for(int jy = results[j].top; jy < results[j].bottom;jy++){
				for(int jx = results[j].left;jx < results[j].right;jx++){
					trainset[nBoosted+facecount].data[jy-results[j].top][jx-results[j].left] = original.data[jy][jx];
				}
			}
			logfile << "(" << results[j].top << ","<<results[j].bottom <<","<< results[j].left << ","<<results[j].right << ");"<<flush;
			trainset[nBoosted+facecount].label = 0;labels[nBoosted+facecount] = -1;
			trainset[nBoosted+facecount].CalcgradientImage();
			trainset[nBoosted+facecount].calcintegralImage();
			trainset[nBoosted+facecount].cleartmp();

			nBoosted++;
		}

		indexBoosted++;
	}

	if(nBoosted == facecount){
		GetFeatureSet();
		return 1;
	}else if (indexBoosted == negative_validcount){
		return 0;
	}


}
void CascadeClassifier::TestOnImageSet()
{
	int cnts;
	vector<string> images2test;
	char szPath[MAX_PATH]; 
	CString str;
	BROWSEINFO info;
	FDImage t_img;
	ofstream testLog("info/testlog.txt");

	info.hwndOwner = AfxGetMainWnd()->m_hWnd;  	info.pidlRoot = NULL;   
	info.pszDisplayName = szPath;   	info.lpszTitle = "请选择测试集所在的文件夹(不含子文件夹）：";   
	info.ulFlags = 0;   	info.lpfn = NULL;   	info.lParam = 0;   	info.iImage = 0;   
	LPITEMIDLIST lp = SHBrowseForFolder(&info);

	if(lp && SHGetPathFromIDList(lp, szPath))	{
		string pathname(szPath);
		AfxMessageBox(pathname.c_str()); 
		int t = dir_file_list(pathname,images2test);
		if(t == 1){
			cnts = images2test.size();
			str.Format(" %s\n共 %d 个文件", szPath,cnts);
			AfxMessageBox(str); 
			testLog << str << endl;
		}
	}
	else{   
		AfxMessageBox("无效的目录，请重新选择");   
		return;
	}
	AfxMessageBox("Ok 确认开始！");
	int detCnt;
	for(int i = 0; i < cnts; i++){
		t_img.Clear();ReadOneTrainingSamplefromfile(images2test[i],t_img);
		detCnt = FaceDetectWithRet(t_img,images2test[i]);
		testLog << images2test[i] << "   " << detCnt << endl << flush;
	}
	return ;
}
void CascadeClassifier::InitTrain()
{
	
	trainlogfile.open(train_log_filename);

	int ret = ReadTrainSet();
	if(ret == 0){
		AfxMessageBox("NO training images");
		return;
	}else if(ret == 1){
		AfxMessageBox("NO enough training images");
		return;
	}
	GeneratePatches2();

	trainlogfile << "InitTrain finished!" <<endl;
	trainlogfile << "facecount:" << facecount << " nonfacecount:" << nonfacecount << endl ;
	trainlogfile << "gFPR: " << gFPR << " gHitrate: " <<gHitrate<< endl;
	trainlogfile << "negative_validcount: " << negative_validcount << " positive_validcount: " << positive_validcount << endl;
	trainlogfile << "boosting_positive:" <<  boosting_positive << "   boosting_negative: " <<boosting_negative << endl;
	
	for(int i = 0; i < totalcount;i++){
		trainset[i].CalcgradientImage();
		trainset[i].calcintegralImage();
		trainset[i].cleartmp();
	}

	GetFeatureSet();
	trainCascadeClassifier();
}

void CascadeClassifier::trainCascadeClassifier()
{
	// over all FPR
	double Ft = gFPR;
	// minimum hit-rate per stage
	double dmin = gHitrate;


	int nPositive = gTotalX_Plus;
	int nNegative = nPositive;
	int nSample = nPositive+nNegative;
	trainset_Cur = new int[nSample];

	ofstream fclassifier(cascade_filename);
	ofstream negative_imageused("info/image_negative.txt");
	ofstream positive_imageused("info/image_positive.txt");
	ofstream trainning_imageused("info/image_trainning.txt");
	rocfile.open("info/roccurve.txt");
	double F_cur = 1.0,d_cur = 1.0;
	double f= 1.0,d = 1.0;

	set<int> used;double ndiff = 0.0;
	int nrounds = 0;
	AdaBoostClassifier tstageClassifier;
	for(int j = 0; j < nPositive; j++)
		trainset_Cur[j] = j;
	for(int j = 0; j < nNegative; j++)
		trainset_Cur[nPositive+j] = facecount+j;
	
	trainlogfile << " Start training " << endl  << flush;

	int i = 0;
	while(F_cur > Ft){
		rocfile << "stage: " << i+1 << endl;
		trainning_imageused <<endl <<   "stage: " << i+1 << endl;
		trainlogfile << " Stage#:" << i+1 << endl;
		trainlogfile << " adding one adaboostclassifier:" << endl; 
		
		tstageClassifier.TrainAdaBoost(nof[i],nPositive,nNegative,dmin,f,d,trainning_imageused);

		strongClassifiers.push_back(tstageClassifier);
		WriteToFile(fclassifier);

		F_cur *= f;d_cur *= d;
		trainlogfile << "sucess!";
		trainlogfile << "F_cur: " << F_cur << " D_cur: " << d_cur << endl << flush;
		trainlogfile << (F_cur > Ft)  << endl << flush;
		if(F_cur > Ft){
			if (gBoostedp){
				trainlogfile << "Boosting Positive sample" << endl << flush;
				positive_imageused << endl <<  "stage: " << i+1 << endl;
				BoostingPositiveSet(positive_imageused);
			}
			trainlogfile << " Boosting Negative sample" << endl << flush;
			negative_imageused << endl <<  "stage: " << i+2 << endl;
			int ret = BoostingNegativeSet(negative_imageused);
			if(ret == 0){
				trainlogfile << endl << "boosting files exhausted! training stop;" << endl << flush;
				break;
			}
			i++;
		}else{
			trainlogfile << "F_cur < Ft : train finished!" << endl << flush;
			break;			
		}
	}

	trainlogfile << "Done : train finished!" << endl << flush;
	WriteToFile(fclassifier);
	fclassifier.close();
	negative_imageused.close();
	positive_imageused.close();
	trainning_imageused.close();
	trainlogfile.close();
	rocfile.close();
	AfxMessageBox("Train Finished!");

}