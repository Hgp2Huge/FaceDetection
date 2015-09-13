typedef struct Feature{
	double fea[32];
}Feature;


struct AdaBoostClassifier
{
	int count;
	vector<WeakClassifier> weakClassifiers;
	double thresh;
	vector<int> trainSet_sampled;

	AdaBoostClassifier();
	~AdaBoostClassifier();
	void Clear(void);
	AdaBoostClassifier& operator=(const AdaBoostClassifier& source);
	void InitToGivenSize(const int size);
	void GetThresh();

	void WriteToFile(ofstream& f);
	void ReadFromFile(ifstream& f);

	inline REAL GetValue( FDImage& im) ;
	inline int Apply( FDImage& im) ;

	void TrainAdaBoost(int rounds,const int nPositive,const int nNegative,double dmin,double &f_cur,double &d_cur,ofstream& fileused);
	double CalcAUCscore(double dmin,double &f_cur,double &d_cur,const int nPositive,const int nNegative);
	double CalcAUCscore(int nPositive,int nNegative);
	double CalcAUCscore();
	double Trap_Area(double x1,double x2,double y1,double y2);
	void sampleTrainSet(int * trainset_Cur,double * weights,int n,int nSampled,ofstream& fileused);
	void sampleTrainFea(vector<Feature>& xfea,int nSampled);
	//void UpdateandNormalizeWeights();
};

REAL AdaBoostClassifier::GetValue(FDImage& im) 
{
	UINT i;
	REAL value;

	value = 0.0;
//	trainlogfile << " GetValue: " << weakClassifiers.size() << " values :";
	for(i=0;i<weakClassifiers.size();i++){
		value += weakClassifiers[i].Apply(im);
	//	trainlogfile << value << " ";
	}
	//trainlogfile << " ret :" << value/weakClassifiers.size() << endl; 
	return value/weakClassifiers.size();
}

int AdaBoostClassifier::Apply(FDImage& im) 
{
	return (GetValue(im)>=thresh)?1:0;
}