
// FaceDetection.h : FaceDetection Ӧ�ó������ͷ�ļ�
//
#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"       // ������


// CFaceDetectionApp:
// �йش����ʵ�֣������ FaceDetection.cpp
//

class CFaceDetectionApp : public CWinApp
{
public:
	CFaceDetectionApp();


// ��д
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// ʵ��
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
	afx_msg void OnTrainStart();
	afx_msg void OnTestStart();
};

extern CFaceDetectionApp theApp;
