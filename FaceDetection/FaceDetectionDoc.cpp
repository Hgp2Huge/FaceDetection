
// FaceDetectionDoc.cpp : CFaceDetectionDoc ���ʵ��
//

#include "stdafx.h"

// SHARED_HANDLERS ������ʵ��Ԥ��������ͼ������ɸѡ�������
// ATL ��Ŀ�н��ж��壬�����������Ŀ�����ĵ����롣
#ifndef SHARED_HANDLERS
#include "FaceDetection.h"
#endif
#include <vector>
#include <fstream>
#include <math.h>
#include <mmsystem.h>
using namespace std;
#include "FDImage.h"
#include "WeakClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"
#include "FaceDetectionDoc.h"

#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CFaceDetectionDoc

IMPLEMENT_DYNCREATE(CFaceDetectionDoc, CDocument)

BEGIN_MESSAGE_MAP(CFaceDetectionDoc, CDocument)
//	ON_COMMAND(ID_FILE_OPEN, &CFaceDetectionDoc::OnFileOpen)
END_MESSAGE_MAP()


// CFaceDetectionDoc ����/����

CFaceDetectionDoc::CFaceDetectionDoc()
{
	// TODO: �ڴ����һ���Թ������

}

CFaceDetectionDoc::~CFaceDetectionDoc()
{
}

BOOL CFaceDetectionDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: �ڴ�������³�ʼ������
	// (SDI �ĵ������ø��ĵ�)

	return TRUE;
}




// CFaceDetectionDoc ���л�

void CFaceDetectionDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: �ڴ���Ӵ洢����
	}
	else
	{
		// TODO: �ڴ���Ӽ��ش���
	}
}

#ifdef SHARED_HANDLERS

// ����ͼ��֧��
void CFaceDetectionDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// �޸Ĵ˴����Ի����ĵ�����
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// ������������֧��
void CFaceDetectionDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// ���ĵ����������������ݡ�
	// ���ݲ���Ӧ�ɡ�;���ָ�

	// ����:  strSearchContent = _T("point;rectangle;circle;ole object;")��
	SetSearchContent(strSearchContent);
}

void CFaceDetectionDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = NULL;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != NULL)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CFaceDetectionDoc ���

#ifdef _DEBUG
void CFaceDetectionDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CFaceDetectionDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CFaceDetectionDoc ����


//void CFaceDetectionDoc::OnFileOpen()
//{
//	// TODO: �ڴ���������������
//
//}


BOOL CFaceDetectionDoc::OnOpenDocument(LPCTSTR lpszPathName)
{
	if (!CDocument::OnOpenDocument(lpszPathName))
		return FALSE;

	// TODO:  �ڴ������ר�õĴ�������
	BeginWaitCursor();
//	trainlogfile.open(train_log_filename);
	image.Clear();
	image.Load(string((char*)lpszPathName));
//	image.CalcgradientImage();
//	image.calcintegralImage();
	UpdateAllViews(NULL);
	EndWaitCursor();
	return TRUE;
}
