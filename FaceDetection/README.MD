================================================================================  
#FaceDetection 项目概述  
===============================================================================  
###VS2010   

FaceDetection.vcxproj  
FaceDetection.vcxproj.filters  
FaceDetection.h  
FaceDetection.cpp  
FaceDetection.rc  
res\FaceDetection.ico  
res\FaceDetection.rc2  


FaceDetectionDoc.h，FaceDetectionDoc.cpp - 文档  
FaceDetectionView.h，FaceDetectionView.cpp - 文档的视图  

StdAfx.h，StdAfx.cpp  
这些文件用于生成名为 FaceDetection.pch 的预编译头 (PCH) 文件和名为 StdAfx.obj 的预编译类型文件。  
Resource.h  
这是标准头文件，它定义新的资源 ID。  
Microsoft Visual C++ 读取并更新此文件。
FaceDetection.manifest

/////////////////////////////////////////////////////////////////////////////
###Global.cpp Global.h  
This two files inlude utility function and global parameters.   
e.g.  
// Initialize global data  
void ClearUpGlobalData(void);  
void InitGlobalData(void);  

// weak classifier training   
void trainLogistic(vector<Feature>& xfea,vector<int> & ylabel,vector<WeakClassifier>& hk,int totalx);  

// generate SURF local features  
void GeneratePatches();  
void GeneratePatches2();  
void GetFeatureSet();  
void getFeatureDescriptor(FDImage &im,double * pFea,int x,int y,int rh,int rw);  
void getFeatureDescriptor(FDImage &im,double * pFea,int x,int y,int rh,int rw,double ratio);  
void getRectFeatureDescriptor(FDImage &im,double* pFea,int p1,int p2,int p3,int p4);  
/////////////////////////////////////////////////////////////////////////////  
###FDImage  
This class is used to save image in memory and compute gradient image and interal image  

/////////////////////////////////////////////////////////////////////////////  
###CascadeClassifier  
This class implements function to train a cascade strong classifier (AdaboostClassifier).  

/////////////////////////////////////////////////////////////////////////////  
###AdaboostClassifier  
This class implements function to train number of weak classifiers (WeakClassifier).  

/////////////////////////////////////////////////////////////////////////////  
###WeakClassifier  
This class implements the weak classifier.

/////////////////////////////////////////////////////////////////////////////  
###Feature  
This struct is used to store the 32-D SURF feature.  

  
