#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double REAL;
#else
typedef float REAL;
#endif

typedef struct gradientData{
	double d[8];
}gradientData;

typedef struct F32Data{
	double	f[8];
}F32Data;

class FDImage
{
public:
	FDImage();
	~FDImage();

	void Clear(void); 
	void SetSize(const CSize size);
	FDImage& operator=(const FDImage& source);
	void Resize(FDImage &result,  REAL ratio) const;
	void Copy(const FDImage& source);
	void Load(const string& filename);
	void Save(const string& filename) const;
	void cleartmp();

	void CalcgradientImage();
	void calcintegralImage();
	void FDImage::CalcSquareAndIntegral(FDImage& square, FDImage& image) const;

public:
	int height; // height, or, number of rows of the image
	int width;  // width, or, number of columns of the image
	REAL** data;  // auxiliary pointers to accelerate the read/write of the image
	// no memory is really allocated, use memory in (buf)
	// data[i][j] is a pixel's gray value in (i)th row and (j)th column
	REAL* buf;    // pointer to a block of continuous memory containing the image
	int label;
	REAL variance;

	gradientData* m_gradientImage;
	F32Data* m_integralIamge;
};