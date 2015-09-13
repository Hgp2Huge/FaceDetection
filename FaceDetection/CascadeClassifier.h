struct CascadeClassifier
{
	int count;
	vector<AdaBoostClassifier> strongClassifiers;
	int indexBoosted;
	CascadeClassifier();
	~CascadeClassifier();
	void Clear(void);
	CascadeClassifier& operator=(const CascadeClassifier& source);

	void InitTrain();
	void WriteToFile(ofstream& f);
	void ReadFromFile(ifstream& f);
	void LoadDefaultCascade(void);

	void trainCascadeClassifier();
	virtual void FaceDetect_ScaleImage(FDImage& original/*,const CString filename*/) ;
	virtual void FaceDetect_ScaleTempl(FDImage& original/*,const CString filename*/) ;
	virtual int  FaceDetectWithRet(FDImage& original,const string filename) ;
	double Apply(FDImage &im);
	double ApplyValue(FDImage &im,double dif);

	int  BoostingPositiveSet(ofstream &logfile);
	int  BoostingNegativeSet(ofstream &logfile);

	void DrawResults(FDImage& image, const vector<CRect>& results) const;
    void PostProcess(vector<CRect>& result, const int combine_min);	

	void TestOnImageSet();
};

