
// FaceDetectionView.cpp : CFaceDetectionView 类的实现
//

#include "stdafx.h"
#include <vector>
using namespace std;
#include "FDImage.h"
#include "WeakClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"
#include <time.h>
#include <stdio.h>
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "FaceDetection.h"
#endif

#include "FaceDetectionDoc.h"
#include "FaceDetectionView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CFaceDetectionView

IMPLEMENT_DYNCREATE(CFaceDetectionView, CView)

BEGIN_MESSAGE_MAP(CFaceDetectionView, CView)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DX1, &CFaceDetectionView::OnTestgradientimageDx1)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DX2, &CFaceDetectionView::OnTestgradientimageDx2)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DY1, &CFaceDetectionView::OnTestgradientimageDy1)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DY2, &CFaceDetectionView::OnTestgradientimageDy2)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DU1, &CFaceDetectionView::OnTestgradientimageDu1)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DU2, &CFaceDetectionView::OnTestgradientimageDu2)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DV1, &CFaceDetectionView::OnTestgradientimageDv1)
	ON_COMMAND(ID_TESTGRADIENTIMAGE_DV2, &CFaceDetectionView::OnTestgradientimageDv2)
//	ON_COMMAND(ID_FACEDETECTION_HAAR, &CFaceDetectionView::OnFacedetectionHaar)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DX1, &CFaceDetectionView::OnTestintegralimageDx1)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DX2, &CFaceDetectionView::OnTestintegralimageDx2)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DY1, &CFaceDetectionView::OnTestintegralimageDy1)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DY2, &CFaceDetectionView::OnTestintegralimageDy2)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DU1, &CFaceDetectionView::OnTestintegralimageDu1)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DU2, &CFaceDetectionView::OnTestintegralimageDu2)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DV1, &CFaceDetectionView::OnTestintegralimageDv1)
	ON_COMMAND(ID_TESTINTEGRALIMAGE_DV2, &CFaceDetectionView::OnTestintegralimageDv2)
//	ON_COMMAND(ID_FACEDETECTION_SCALE, &CFaceDetectionView::OnFacedetectionScale)

	ON_COMMAND(ID_FACEDETECTION_IMAGESCALE, &CFaceDetectionView::OnFacedetectionImagescale)
	ON_COMMAND(ID_FACEDETECTION_TEMPLATESCALE, &CFaceDetectionView::OnFacedetectionTemplatescale)
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

// CFaceDetectionView 构造/析构

CFaceDetectionView::CFaceDetectionView()
{
	// TODO: 在此处添加构造代码

}

CFaceDetectionView::~CFaceDetectionView()
{
}

BOOL CFaceDetectionView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CView::PreCreateWindow(cs);
}

// CFaceDetectionView 绘制

void CFaceDetectionView::OnDraw(CDC* pDC)
{
	CFaceDetectionDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
	int i,j;
	FDImage& im = pDoc->image;
	CRect   clientWH;
	CDC MemDC; 
	CBitmap MemBitmap;

	MemDC.CreateCompatibleDC(NULL); 

	GetClientRect(clientWH);
	MemBitmap.CreateCompatibleBitmap(pDC,clientWH.Width(),clientWH.Height()); 

	CBitmap *pOldBit=MemDC.SelectObject(&MemBitmap); 
	MemDC.FillSolidRect(0,0,clientWH.Width(),clientWH.Height(),RGB(255,255,255)); 
	if(im.height>0){
		for(i=0;i<im.height;i++)
			for(j=0;j<im.width;j++)
				MemDC.SetPixel(j,i,RGB(im.data[i][j],im.data[i][j],im.data[i][j]));
	}

	pDC->BitBlt(0,0,clientWH.Width(),clientWH.Height(),&MemDC,0,0,SRCCOPY); 

	MemBitmap.DeleteObject(); 
	MemDC.DeleteDC(); 

	//if(im.height>0){
	//	for(i=0;i<im.height;i++)
	//		for(j=0;j<im.width;j++)
	//			pDC->SetPixel(j,i,RGB(im.data[i][j],im.data[i][j],im.data[i][j]));
	//}
	// TODO: 在此处为本机数据添加绘制代码
}


// CFaceDetectionView 诊断

#ifdef _DEBUG
void CFaceDetectionView::AssertValid() const
{
	CView::AssertValid();
}

void CFaceDetectionView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CFaceDetectionDoc* CFaceDetectionView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CFaceDetectionDoc)));
	return (CFaceDetectionDoc*)m_pDocument;
}
#endif //_DEBUG


// CFaceDetectionView 消息处理程序

void CFaceDetectionView::Test_gradientimage(int Index)
{
	CFaceDetectionDoc* pDoc = GetDocument();
	CDC* pDC = GetDC();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
	int i,j;
	FDImage im;im.Copy(pDoc->image);
	im.CalcgradientImage();
	if(im.height>0){
		for(i=0;i<im.height;i++)
			for(j=0;j<im.width;j++)
				pDC->SetPixel(j,i,RGB(im.m_gradientImage[i*im.width+j].d[Index],im.m_gradientImage[i*im.width+j].d[Index],im.m_gradientImage[i*im.width+j].d[Index]));
	}
	im.Clear();
}
void CFaceDetectionView::OnTestgradientimageDx1()
{
	Test_gradientimage(0);
}
void CFaceDetectionView::OnTestgradientimageDx2()
{
	Test_gradientimage(1);
}
void CFaceDetectionView::OnTestgradientimageDy1()
{
	Test_gradientimage(2);
}
void CFaceDetectionView::OnTestgradientimageDy2()
{
	Test_gradientimage(3);
}
void CFaceDetectionView::OnTestgradientimageDu1()
{
	Test_gradientimage(4);
}
void CFaceDetectionView::OnTestgradientimageDu2()
{
	Test_gradientimage(5);
}
void CFaceDetectionView::OnTestgradientimageDv1()
{
	Test_gradientimage(6);
}
void CFaceDetectionView::OnTestgradientimageDv2()
{
	Test_gradientimage(7);
}

///////////////////////////////////////////////////////////////////////////////////////////////


void CFaceDetectionView::Test_integralimage(int Index)
{
	CFaceDetectionDoc* pDoc = GetDocument();
	CDC* pDC = GetDC();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
	int i,j;
	FDImage im;im.Copy(pDoc->image);
	im.CalcgradientImage();im.calcintegralImage();im.cleartmp();
	if(im.height>0)	{
		for(i=0;i<im.height+1;i++)
			for(j=0;j<im.width+1;j++)
				pDC->SetPixel(j,i,RGB(im.m_integralIamge[i*(im.width+1)+j].f[Index],im.m_integralIamge[i*(im.width+1)+j].f[Index],im.m_integralIamge[i*(im.width+1)+j].f[Index]));
	}
	im.Clear();
}
void CFaceDetectionView::OnTestintegralimageDx1()
{
	Test_integralimage(0);
}
void CFaceDetectionView::OnTestintegralimageDx2()
{
	Test_integralimage(1);

}
void CFaceDetectionView::OnTestintegralimageDy1()
{
	Test_integralimage(2);
}
void CFaceDetectionView::OnTestintegralimageDy2()
{
	Test_integralimage(3);

}
void CFaceDetectionView::OnTestintegralimageDu1()
{
	Test_integralimage(4);
}
void CFaceDetectionView::OnTestintegralimageDu2()
{
	Test_integralimage(5);
}
void CFaceDetectionView::OnTestintegralimageDv1()
{
	Test_integralimage(6);
}
void CFaceDetectionView::OnTestintegralimageDv2()
{
	Test_integralimage(7);
}


//void CFaceDetectionView::OnFacedetectionScale()
//{
//	CFaceDetectionDoc* pDoc = GetDocument();
//	CDC* pDC = GetDC();
//	ASSERT_VALID(pDoc);
//	if (!pDoc)
//		return;
//	cascade->LoadDefaultCascade();
//	FDImage &im = pDoc->image;
//	if(cascade->count>0)
//	{
//		cascade->FaceDetect2(im);
//		Invalidate(FALSE);
//	}
//}

long long getTickCount()
{
#if defined WIN32 || defined WIN64 || defined _WIN64
	LARGE_INTEGER counter;
	QueryPerformanceCounter( &counter );
	return (long long)counter.QuadPart;
#elif defined __linux || defined __linux__
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (long long)tp.tv_sec*1000000000 + tp.tv_nsec;
#else    
	struct timeval tv;
	struct timezone tz;
	gettimeofday( &tv, &tz );
	return (long long)tv.tv_sec*1000000 + tv.tv_usec;
#endif
}

double getTickFrequency()
{
#if defined WIN32 || defined WIN64 || defined _WIN64
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return (double)freq.QuadPart;
#elif defined __linux || defined __linux__
	return 1e9;
#else
	return 1e6;
#endif
}



void CFaceDetectionView::OnFacedetectionImagescale()
{

	BeginWaitCursor();
	CFaceDetectionDoc* pDoc = GetDocument();
	CDC* pDC = GetDC();
	long long t1, t2;
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
	cascade->LoadDefaultCascade();
	if(cascade ==NULL)
		return;
	FDImage &im = pDoc->image;
	if(cascade->count>0)
	{
		t1 = getTickCount();
		cascade->FaceDetect_ScaleImage(im);
		t2 = getTickCount();
		double duration = ((double)(t2 - t1))/(getTickFrequency()*1e-3);	
		char a[30];
		sprintf(a,"Finished in %.4f msec!\n", duration);
		AfxMessageBox(a);
		Invalidate(FALSE);
	}
	EndWaitCursor();
}


void CFaceDetectionView::OnFacedetectionTemplatescale()
{
	BeginWaitCursor();
	CFaceDetectionDoc* pDoc = GetDocument();
	CDC* pDC = GetDC();
	ASSERT_VALID(pDoc);
	long long t1, t2;
	if (!pDoc)
		return;
	cascade->LoadDefaultCascade();
	FDImage &im = pDoc->image;
	if(cascade->count>0)
	{
		t1 = getTickCount();
		cascade->FaceDetect_ScaleTempl(im);
		t2 = getTickCount();
		double duration = ((double)(t2 - t1))/(getTickFrequency()*1e-3);
		char a[30];
		sprintf(a,"Finished in %.4f msec!\n", duration);
		AfxMessageBox(a);
		Invalidate(FALSE);
	}
	EndWaitCursor();
}


BOOL CFaceDetectionView::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	return TRUE;
	//return CView::OnEraseBkgnd(pDC);
}
