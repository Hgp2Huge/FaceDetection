
#pragma   comment(lib,   "vfw32.lib ")
#pragma comment (lib , "comctl32.lib")

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <fstream>
using namespace std;

#include "FDImage.h"
#include "WeakClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

FDImage::FDImage():height(0),width(0),data(NULL),label(-1),m_integralIamge(NULL),m_gradientImage(NULL),variance(0.0)
{

}

FDImage::~FDImage()
{
	Clear();
}

void FDImage::Clear(void)
{

	if(data == NULL)
		ASSERT(buf == NULL);
	else
	{
		ASSERT(buf != NULL);
		for(int i=0;i<height;i++){
			data[i] = NULL;
		}
		delete[] data;	data = NULL;
		delete[] buf;  	buf = NULL;
	
		height = width = 0;
		variance = 0.0;
		label = -1;
	}	
	delete [] m_gradientImage;m_gradientImage=NULL;
	delete [] m_integralIamge;m_integralIamge=NULL;
	//if(data  != NULL){
	//	for(int i=0;i<height;i++)	data[i] = NULL;
	//	delete[] data;	data = NULL;
	//	delete[] buf;	buf	= NULL;
	//}
	//if(m_gradientImage != NULL){
	//	delete [] m_gradientImage;
	//	m_gradientImage=NULL;
	//}
	//if(m_integralIamge != NULL){
	//	delete [] m_integralIamge;
	//	m_gradientImage=NULL;
	//}
	//height = width = 0;
	//variance = 0.0;
	//label = -1;

}

void FDImage::cleartmp()
{
	if(data == NULL)
		ASSERT(buf == NULL);
	else
	{
		ASSERT(buf != NULL);
		for(int i=0;i<height;i++){
			data[i] = NULL;
		}
		delete[] data;	data = NULL;
		delete[] buf;  	buf = NULL;
		delete [] m_gradientImage;m_gradientImage=NULL;
	}
	//if(data  != NULL){
	//	for(int i=0;i<height;i++)	data[i] = NULL;
	//	delete[] data;	data = NULL;
	//	delete[] buf;	buf = NULL;
	//}
	//if(m_gradientImage != NULL){
	//	delete [] m_gradientImage;
	//	m_gradientImage=NULL;
	//}
}
void FDImage::Copy(const FDImage& source)
	// the ONLY way to make a copy of 'source' to this image
{
	ASSERT(source.height > 0);
	ASSERT(source.width > 0);
	if(&source == this)	return;
	SetSize(CSize(source.height,source.width));
	label = source.label;
	memcpy(buf,source.buf,sizeof(REAL)*height*width);
}

void FDImage::Load(const string& filename)
{
	IplImage* img;

	img = cvLoadImage(filename.c_str(),0);
	SetSize(CSize(img->height,img->width));
	for(int i=0,ih=img->height,iw=img->width;i<ih;i++)
	{
		REAL* pdata = data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
		for(int j=0;j<iw;j++) pdata[j] = pimg[j];
	}

	cvReleaseImage(&img);
}

void FDImage::Save(const string& filename) const
{
	IplImage* img;

	img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	for(int i=0,ih=img->height,iw=img->width;i<ih;i++)
	{
		REAL* pdata = data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
		for(int j=0;j<iw;j++) pimg[j] = (unsigned char)pdata[j];
	}
	cvSaveImage(filename.c_str(),img);
	cvReleaseImage(&img);	
}

void FDImage::SetSize(const CSize size)
	// 'size' is the new size of the image, if necessary, memory is reallocated
	// size.cx is the new height and size.cy is the new width
{
	if((size.cx == height) && (size.cy == width) && (buf != NULL) &&(data != NULL) ) return; 
	ASSERT(size.cx >= 0); ASSERT(size.cy >= 0);Clear();
	height = size.cx;	width = size.cy;
	buf = new REAL[height*width]; ASSERT(buf != NULL);
	data = new REAL*[height];	ASSERT(data != NULL);
	for(int i=0;i<height;i++)	data[i] = &buf[i*width];

}

FDImage& FDImage::operator=(const FDImage& source)
{
	SetSize(CSize(source.height,source.width));
	memcpy(m_integralIamge,source.m_integralIamge,sizeof(double)*height*width);
	//memcpy(data,source.data,sizeof(double)*height*width);
	label = source.label;

	return *this;
}

void FDImage::Resize(FDImage &result, REAL ratio) const
{
	result.SetSize(CSize(int(height*ratio),int(width*ratio)));
	ratio = 1/ratio;
	for(int i=0,rh=result.height,rw=result.width;i<rh;i++)
		for(int j=0;j<rw;j++) {
			int x0,y0;
			REAL x,y,fx0,fx1;
			x = j*ratio; y = i*ratio;
			x0 = (int)(x);
			y0 = (int)(y);

			//by Jianxin Wu  
			//1. The conversion of float to int in C is towards to 0 point, i.e. the floor function for positive numbers, and ceiling function for negative numbers.
			//2. We only make use of ratio<1 in this applicaiton, and all numbers involved are positive.
			//Using these, we have 0<=x<=height-1 and 0<=y<=width-1. Thus, boundary conditions check is not necessary.
			//In languages other than C/C++ or ratio>=1, take care. 
			if (x0 == width-1) x0--;
			if (y0 == height-1) y0--;

			x = x - x0; y = y - y0;

			fx0 = data[y0][x0] + x*(data[y0][x0+1]-data[y0][x0]);
			fx1 = data[y0+1][x0] + x*(data[y0+1][x0+1]-data[y0+1][x0]);

			result.data[i][j] = fx0 + y*(fx1-fx0);
		}
}

void FDImage::CalcgradientImage()
{
	int y,x;
	m_gradientImage = new gradientData[height*width];
	double gradient = 0;
	for(x =0 ; x< width; x++)
		for(int j = 0; j < 8; j++){
			m_gradientImage[x].d[j] = 0;
			m_gradientImage[(height-1)*width + x].d[j] = 0;
		}
	for(y =0 ; y< height; y++)
		for(int j = 0; j < 8; j++){
			m_gradientImage[y*width].d[j] = 0;
			m_gradientImage[y*width+width-1].d[j] = 0;
		}
	for(y = 1;y <height-1; y++)
		for(x = 1; x < width-1; x++){
			//dx
			gradient = data[y][x+1] - data[y][x-1];
			m_gradientImage[y*width+x].d[0] = abs(gradient) - gradient;
			m_gradientImage[y*width+x].d[1] = abs(gradient) + gradient;
			//dy
			gradient = data[y+1][x] - data[y-1][x];
			m_gradientImage[y*width+x].d[2] = abs(gradient) - gradient;
			m_gradientImage[y*width+x].d[3] = abs(gradient) + gradient;
			//du
			gradient = data[y-1][x+1] - data[y+1][x-1];
			m_gradientImage[y*width+x].d[4] = abs(gradient) - gradient;
			m_gradientImage[y*width+x].d[5] = abs(gradient) + gradient;
			//dv
			gradient = data[y+1][x+1] - data[y-1][x-1];
			m_gradientImage[y*width+x].d[6] = abs(gradient) - gradient;
			m_gradientImage[y*width+x].d[7] = abs(gradient) + gradient;
		}
	//trainlogfile << "image" << endl;
	//for(int i = 0; i < height; i++){
	//	for(int j = 0; j < width; j++){
	//		trainlogfile << data[i][j]<< ' ';	
	//	}
	//	trainlogfile << endl;
	//}
	//trainlogfile << "gradientimage1" << endl;
	//for(int i = 0; i < height; i++){
	//	for(int j = 0; j < width; j++){
	//		trainlogfile << m_gradientImage[i*width+j].d[0] << ' ';	
	//	}
	//	trainlogfile << endl;
	//}
	//trainlogfile << "gradientimage2" << endl;
	//for(int i = 0; i < height; i++){
	//	for(int j = 0; j < width; j++){
	//		trainlogfile << m_gradientImage[i*width+j].d[1] << ' ';	
	//	}
	//	trainlogfile << endl;
	//}
}

void FDImage::calcintegralImage()
{
	m_integralIamge = new F32Data[(height+1)*(width+1)];
	double partialsum[8];
	for(int i =0; i < 8; i++) partialsum[i] = 0;
	// add one row 
	for(int k = 0; k < 8; k++){
		for(int x = 0; x < width+1 ; x++){
			m_integralIamge[x].f[k] = 0;
		}
	}
	//add one col
	for(int k = 0; k < 8; k++){
		for(int y = 0; y < height+1 ; y++){
			m_integralIamge[y*(width+1)].f[k] = 0;
		}
	}

	for(int y = 1; y < height+1 ;y++){
		for(int i =0; i < 8; i++) partialsum[i] = 0;
		for(int x = 1; x < width+1; x++){
			//Sum: /dx/-dx,/dx/+dx,/dy/-dy,/dy/+dy,/du/-du,/du/+du,/dv/-dv,/dv/+dv
			for(int k = 0; k < 8; k++){
				partialsum[k] += m_gradientImage[(y-1)*width+x-1].d[k];
				m_integralIamge[y*(width+1)+x].f[k] = m_integralIamge[(y-1)*(width+1)+x].f[k] + partialsum[k];
			}
		}
	}
	//trainlogfile << "integralimage" << endl;
	//for(int i = 0; i < height+1; i++){
	//	for(int j = 0; j < width+1; j++){
	//		trainlogfile << m_integralIamge[i*(width+1)+j].f[0] << ' ';	
	//	}
	//	trainlogfile << endl;
	//}
	//trainlogfile << endl;trainlogfile << endl;
}

void FDImage::CalcSquareAndIntegral(FDImage& square, FDImage& image) const
{
	REAL partialsum,partialsum2;

	square.SetSize(CSize(height+1,width+1));
	image.SetSize(CSize(height+1,width+1));

	for(int i=0;i<width+1;i++) square.buf[i]=image.buf[i]=0;
	for(int i=1;i<height+1;i++)
	{
		partialsum = partialsum2 = 0;
		square.data[i][0] = 0;
		image.data[i][0] = 0;
		for(int j=1;j<=width;j++)
		{
			partialsum += (data[i-1][j-1]*data[i-1][j-1]);
			partialsum2 += data[i-1][j-1];
			square.data[i][j] = square.data[i-1][j] + partialsum;
			image.data[i][j] = image.data[i-1][j] + partialsum2;
		}
	}
}