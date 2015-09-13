
// FaceDetectionView.h : CFaceDetectionView 类的接口
//

#pragma once


class CFaceDetectionView : public CView
{
protected: // 仅从序列化创建
	CFaceDetectionView();
	DECLARE_DYNCREATE(CFaceDetectionView)

// 特性
public:
	CFaceDetectionDoc* GetDocument() const;

// 操作
public:

// 重写
public:
	virtual void OnDraw(CDC* pDC);  // 重写以绘制该视图
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:

// 实现
public:
	virtual ~CFaceDetectionView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
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

#ifndef _DEBUG  // FaceDetectionView.cpp 中的调试版本
inline CFaceDetectionDoc* CFaceDetectionView::GetDocument() const
   { return reinterpret_cast<CFaceDetectionDoc*>(m_pDocument); }
#endif

