
// FaceDetectionView.h : CFaceDetectionView ��Ľӿ�
//

#pragma once


class CFaceDetectionView : public CView
{
protected: // �������л�����
	CFaceDetectionView();
	DECLARE_DYNCREATE(CFaceDetectionView)

// ����
public:
	CFaceDetectionDoc* GetDocument() const;

// ����
public:

// ��д
public:
	virtual void OnDraw(CDC* pDC);  // ��д�Ի��Ƹ���ͼ
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:

// ʵ��
public:
	virtual ~CFaceDetectionView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ���ɵ���Ϣӳ�亯��
protected:
	DECLARE_MESSAGE_MAP()
public:
	void Test_gradientimage(int Index);
	afx_msg void OnTestgradientimageDx1();
	afx_msg void OnTestgradientimageDx2();
	afx_msg void OnTestgradientimageDy1();
	afx_msg void OnTestgradientimageDy2();
	afx_msg void OnTestgradientimageDu1();
	afx_msg void OnTestgradientimageDu2();
	afx_msg void OnTestgradientimageDv1();
	afx_msg void OnTestgradientimageDv2();
//
	afx_msg void Test_integralimage(int Index);
	afx_msg void OnTestintegralimageDx1();
	afx_msg void OnTestintegralimageDx2();
	afx_msg void OnTestintegralimageDy1();
	afx_msg void OnTestintegralimageDy2();
	afx_msg void OnTestintegralimageDu1();
	afx_msg void OnTestintegralimageDu2();
	afx_msg void OnTestintegralimageDv1();
	afx_msg void OnTestintegralimageDv2();
//	

	afx_msg void OnFacedetectionImagescale();
	afx_msg void OnFacedetectionTemplatescale();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
};

#ifndef _DEBUG  // FaceDetectionView.cpp �еĵ��԰汾
inline CFaceDetectionDoc* CFaceDetectionView::GetDocument() const
   { return reinterpret_cast<CFaceDetectionDoc*>(m_pDocument); }
#endif

