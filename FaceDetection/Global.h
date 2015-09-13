struct CascadeClassifier;
struct patch;
class  FDImage;
struct Feature;
struct Weight;
struct WeakClassifier;

extern CString 	option_filename;
extern string cascade_filename;
extern CString train_log_filename;

extern ofstream rocfile;
extern int sx;
extern int sy;
extern int max_nodes;
extern double gFPR;
extern double gHitrate;
extern int gTotalX_Plus;
extern int gsampled;
extern double galpha;
extern double gdelta; 
extern BOOL   gBoostedp; 
extern int    gCombine_min;
extern vector<string> Boosting_positive_file;
extern vector<string> Boosting_negative_file;
extern vector<int> nof;
extern CascadeClassifier* cascade;


extern vector<patch> patches;
extern ofstream trainlogfile;
extern int totalfeatures;
extern int facecount,nonfacecount;
extern int totalcount,negative_validcount,positive_validcount;
extern FDImage* trainset;
extern int* trainset_Cur;
extern BOOL* labels;
extern Feature* FeatureSet;
extern string boosting_positive;
extern string boosting_negative;

void ClearUpGlobalData(void);
void InitGlobalData(void);
void QuickSortDescend(double* values,int* vindex,const int n);
int dir_file_list(string folderPath, vector<string> &file_vec);
int ReadTrainSet();
void ReadOneTrainingSamplefromfile(const string& filename,FDImage& image);

void GeneratePatches();
void GeneratePatches2();
void GetFeatureSet();
void getFeatureDescriptor(FDImage &im,double * pFea,int x,int y,int rh,int rw);
void getFeatureDescriptor(FDImage &im,double * pFea,int x,int y,int rh,int rw,double ratio);
void getRectFeatureDescriptor(FDImage &im,double* pFea,int p1,int p2,int p3,int p4);

double sigmoid(double x);
void trainLogistic(vector<Feature>& xfea,vector<int> & ylabel,vector<WeakClassifier>& hk,int totalx);
void FeatureNormalization(double *fea,double d);