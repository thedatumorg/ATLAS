#include "headers.h"


// -----------------------------------------------------------------------------
//  QALSH: the hash tables of qalsh are indexed by b+ tree. QALSH is used to 
//  solve the problem of high-dimensional c-Approximate Nearest Neighbor (c-ANN)
//  search.
// -----------------------------------------------------------------------------
int read_data2(
    int son, 
	int dim_,
	float* data,
	BlockFile2* d_blockFile)
{
    int blk_pos = ((long long)dim_ * sizeof(float) * (son - 1)) % d_blockFile->blocklength;
    int blk_id = (long long)dim_ * sizeof(float) * (son - 1) / d_blockFile->blocklength;
    char *blk = new char[d_blockFile->blocklength];
    d_blockFile->read_block(blk, blk_id);
    for (int i = 0; i < dim_; ++i){
        if (blk_pos == d_blockFile->blocklength){
            blk_id++;
            d_blockFile->read_block(blk, blk_id);
            blk_pos = 0;
        }
        memcpy(&data[i], &blk[blk_pos], sizeof(float));
        blk_pos += sizeof(float);
    }  
    return 1;	
}


QALSH::QALSH()						// constructor
{
	n_pts_ = dim_ = B_ = -1;
	appr_ratio_ = beta_ = delta_ = -1.0;

	w_ = p1_ = p2_ = alpha_ = -1.0;
	m_ = l_ = -1;

	a_array_ = NULL;
	s_array_ = NULL;
	trees_ = NULL;
}

// -----------------------------------------------------------------------------
QALSH::~QALSH()						// destructor
{
	if (a_array_) {
		delete[] a_array_; a_array_ = NULL;
		g_memory -= SIZEFLOAT * m_ * dim_;
	}

	if (trees_) {
		for (int i = 0; i < m_; i++) {
			delete trees_[i]; trees_[i] = NULL;
		}
		delete[] trees_; trees_ = NULL;
	}
}

// -----------------------------------------------------------------------------
void QALSH::init(					// init params of qalsh
	int   n,							// number of points
	int   m,
//	int   n_ref,
	int   n_ring,
	int   d,							// dimension of space
	int   sub_d,                        // dimension of subspace
	int   B,							// page size
//	float ratio,						// approximation ratio
	char* output_folder)				// folder of info of qalsh
{
	n_pts_ = n;						// init <n_pts_>
	dim_   = d;						// init <dim_>
	B_     = B;						// init <B_>
//	appr_ratio_ = ratio;			// init <appr_ratio_>
	s_dim_ = sub_d;
	m_ = m;
	n_ring_ = n_ring;
	strcpy(index_path_, output_folder);
	strcat(index_path_, "L2_indices/");
	gen_hash_func();				// init <a_array_>     REVISE
	trees_ = NULL;					// init <trees_>
}

// -----------------------------------------------------------------------------
void QALSH::calc_params()		
{
	delta_ = 1.0f / E;
	beta_  = 100.0f / n_pts_;
	w_ = sqrt((8.0f * appr_ratio_ * appr_ratio_ * log(appr_ratio_))
		/ (appr_ratio_ * appr_ratio_ - 1.0f));

	p1_ = calc_l2_prob(w_ / 2.0f);
	p2_ = calc_l2_prob(w_ / (2.0f * appr_ratio_));

	float para1 = sqrt(log(2.0f / beta_));
	float para2 = sqrt(log(1.0f / delta_));
	float para3 = 2.0f * (p1_ - p2_) * (p1_ - p2_);
	
	float eta = para1 / para2;		
	alpha_ = (eta * p1_ + p2_) / (1.0f + eta);

	m_ = (int) ceil((para1 + para2) * (para1 + para2) / para3);
	l_ = (int) ceil((p1_ * para1 + p2_ * para2) * (para1 + para2) / para3);
}

// -----------------------------------------------------------------------------
float QALSH::calc_l2_prob(			
	float x)							
{
	return new_gaussian_prob(x);
}

// -----------------------------------------------------------------------------
void QALSH::display_params()		// display params of qalsh  //不变
{

	printf("Parameters of QALSH (L_2 Distance):\n");

	printf("    n          = %d\n", n_pts_);
	printf("    d          = %d\n", dim_);
	printf("    B          = %d\n", B_);
	printf("    ratio      = %0.2f\n", appr_ratio_);
	printf("    w          = %0.4f\n", w_);
	printf("    p1         = %0.4f\n", p1_);
	printf("    p2         = %0.4f\n", p2_);
	printf("    alpha      = %0.6f\n", alpha_);
	printf("    beta       = %0.6f\n", beta_);
	printf("    delta      = %0.6f\n", delta_);
	printf("    m          = %d\n", m_);
	printf("    l          = %d\n", l_);
	printf("    beta * n   = %d\n", 100);
	printf("    index path = %s\n", index_path_);
    
}

// -----------------------------------------------------------------------------
void QALSH::gen_hash_func()			// generate hash function <a_array>   //不变
{
	int sum = m_ * s_dim_ * dim_;               //
	float temp = 0;

	//g_memory += SIZEFLOAT * sum;
	a_array_ = new float[sum];

	for (int i = 0; i < sum; i++) {
		a_array_[i] = gaussian(0.0f, 1.0f);
	}
	
	sum = m_ * s_dim_;
	s_array_ = new float[sum];
    for (int i = 0; i < sum; i++) {            //for the calculation of angles
	    s_array_[i] = gaussian(0.0f, 1.0f);	
	}
	
	for (int i = 0; i < m_; i++){
		temp = 0;
		for(int j = 0; j < s_dim_; j++){
			temp += s_array_[i*s_dim_+j] * s_array_[i*s_dim_+j];
		}
		temp = sqrt(temp);
		for(int j = 0; j < s_dim_; j++)
		    s_array_[i*s_dim_+j] =  s_array_[i*s_dim_+j]/temp;
	}
}

// -----------------------------------------------------------------------------
int QALSH::bulkload(				// build m b-trees by bulkloading
	char* data_set,
	char* output_folder)						// data set
{
	// -------------------------------------------------------------------------
	//  Check whether the default maximum memory is enough
	// -------------------------------------------------------------------------
	g_memory += sizeof(HashValue) * n_pts_;
	// -------------------------------------------------------------------------
	//  Check whether the directory exists. If the directory does not exist, we
	//  create the directory for each folder.
	// -------------------------------------------------------------------------
#ifdef LINUX_						// create directory under Linux
	int len = (int) strlen(index_path_);
	for (int i = 0; i < len; i++) {
		if (index_path_[i] == '/') {
			char ch = index_path_[i + 1];
			index_path_[i + 1] = '\0';
									// check whether the directory exists
			int ret = access(index_path_, F_OK);
			if (ret != 0) {			// create directory
				ret = mkdir(index_path_, 0755);
				if (ret != 0) {
					printf("Could not create directory %s\n", index_path_);
					error("QALSH::bulkload error\n", true);
				}
			}
			index_path_[i + 1] = ch;
		}
	}
#else								// create directory under Windows
	int len = (int) strlen(index_path_);
	for (int i = 0; i < len; i++) {
		if (index_path_[i] == '/') {
			char ch = index_path_[i + 1];
			index_path_[i + 1] = '\0';
									// check whether the directory exists
			int ret = _access(index_path_, 0);
			if (ret != 0) {			// create directory
				ret = _mkdir(index_path_);
				if (ret != 0) {
					printf("Could not create directory %s\n", index_path_);
					error("QALSH::bulkload() error\n", true);
				}
			}
			index_path_[i + 1] = ch;
		}
	}
#endif
	
	// -------------------------------------------------------------------------
	//  Write the file "para" where the parameters and hash functions are 
	//  stored in it.
	// -------------------------------------------------------------------------
	char fname[200];
	strcpy(fname, index_path_);		// write the "para" file
	strcat(fname, "para");
	if (write_para_file(fname)) return 1;

	float cur_val,min_val;
	int min_id;
	int temp_id;
	
	float sum;
	int* last_p = new int[m_];
	
	int* upp = new int[n_ring_];
	int* low = new int[n_ring_];
	
	HashValue* a_points = new HashValue[n_pts_];
	
	float** proj = new float*[n_pts_];
	for(int i = 0; i < n_pts_; i++){
		proj[i] = new float[s_dim_];
	}
	
	Ring* med = new Ring[n_ring_];	
	int* Inval = new int[n_ring_];
	int count;
	float temp;
	
	int ind;	
	float* ref_p = new float[s_dim_];
    double* ref_p2 = new double[s_dim_];
	string resultFile=output_folder;
	resultFile = resultFile + "result.txt";
	string centerFile=output_folder;
	centerFile = centerFile + "center.txt";
	string rootFile=output_folder;
	rootFile = rootFile + "root.txt";
	FILE* fp1 = fopen(centerFile.c_str(),"w");
	FILE* fp3 = fopen(rootFile.c_str(),"w");
	
	int* Root_I = new int [n_ring_];
	float jj;
	
	std::ifstream in(data_set, std::ios::binary); // open data file 
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
	
	int const_ = 10000000;
    int n_batch;
    int n_num = 0;
	
	float* data = new float[dim_];
    //char* data_temp = new char[dim_];

	for(int i =0; i < m_; i++){
		fprintf(fp3, "%d\n", i); 
        in.seekg(0, std::ios::beg);
	
		for(int j = 0; j < s_dim_; j++){
			ref_p[j] = 0;
            ref_p2[j] = 0;
        }
		
		for(int j = 0; j < n_pts_; j++){
            in.seekg(4, std::ios::cur);	
            in.read((char*)(data), 4 * dim_); // read data

           // for(int j2 = 0; j2 < dim_; j2++)
           //     data[j2] = data_temp[j2];   
			
			
			for(int i3 = 0; i3 <s_dim_; i3++){
				sum = 0;
				for(int i5 = 0; i5 < dim_; i5++){
		            sum += data[i5]*a_array_[i*s_dim_*dim_+i3*dim_+i5];
		        }
			    proj[j][i3] = sum;
                ref_p2[i3] += proj[j][i3];				
			}			
		}
		
		for(int j = 0; j <s_dim_; j++){
		    ref_p[j] = (float)(ref_p2[j] / n_pts_);
		}

		for(int j = 0; j < n_pts_; j++){
            a_points[j].proj_= calc_l2_dist( ref_p, proj[j], s_dim_);
            a_points[j].id_= j + 1;  //id from 1 
		}
		
		for(int j = 0; j < s_dim_; j++){
			fprintf(fp1, "%f ", ref_p[j]);
		}
		fprintf(fp1, "\n");

		qsort(a_points, n_pts_, sizeof(HashValue), HashValueQsortComp);       
			
		for(int j = 0; j < n_ring_; j++){
                        float coeff_ = (float) (j+1) / (float) n_ring_;
			Inval[j] = int(n_pts_* coeff_);
			if(Inval[j] >= n_pts_-1)
				Inval[j] = n_pts_-1;
		}
		
		for(int j = 0; j < n_ring_; j++){
            if(j == 0){
			    med[j].num = Inval[j]+1;
				med[j].lo_val = a_points[0].proj_;
				fprintf(fp1, "%f ", a_points[0].proj_);
				med[j].up_val = a_points[Inval[j]].proj_;
				fprintf(fp1, "%f ", a_points[Inval[j]].proj_);   // before revise. fprintf(fp1, "%f ", a_points[j].proj_);
				continue;	
			}
			
			else{
				if( Inval[j]== Inval[j-1]){
					fprintf(fp1, "%f %f ", -1, -1);
				    med[j].num = 0;
				}
				else{
		            med[j].lo_val = a_points[ Inval[j-1]+1 ].proj_;
				    fprintf(fp1, "%f ", a_points[ Inval[j-1]+1 ].proj_);
				    med[j].up_val = a_points[ Inval[j] ].proj_;
				    fprintf(fp1, "%f ", a_points[ Inval[j] ].proj_);
                    med[j].num = Inval[j]-Inval[j-1];
				}	
			}				
		}
		
		fprintf(fp1, "\n");                    //build tables
		for(int j = 0; j < n_pts_; j++){
            temp_id = a_points[j].id_-1;
            sum = 0;
			temp = 0;
            for(int s = 0; s < s_dim_; s++){
				temp += (proj[temp_id][s] - ref_p[s]) * (proj[temp_id][s] - ref_p[s]);  //normalize proj and s_array
			}	
            temp = sqrt(temp);					
            for(int s = 0; s < s_dim_; s++){
				sum += ( (proj[temp_id][s] - ref_p[s]) * s_array_[s_dim_*i + s] );  //normalize proj and s_array
			}
			
			a_points[j].proj_ = sum / temp;
            sum = 0;
			for(int s = 0; s < s_dim_; s++){
			    if(s == 0)
				{sum += ( (proj[temp_id][s] - ref_p[s]) / s_array_[s_dim_*i + s] );}
				else {sum += ( (-1) * (proj[temp_id][s] - ref_p[s]) / s_array_[s_dim_*i + s] );}
			}
		    if(sum < 0){
				a_points[j].proj_ = 2 - a_points[j].proj_;
			}	
		}		

		get_tree_filename( i, fname);
		BTree *bt = new BTree();
		bt->init(fname, B_, n_ring_);
		if (bt->bulkload(a_points, med, i, Root_I, &(last_p[i]), upp, low)) {
			return 1;
		}

		delete bt; bt = NULL;
		
		for(int j = 0; j < n_ring_; j++){
			fprintf(fp3, "%d ", Root_I[j]);
		}
		
	    fprintf(fp3, "%d\n", last_p[i]);
		for(int j = 0; j < n_ring_; j++){
			fprintf(fp3, "%d ", low[j]);
			fprintf(fp3, "%d ", upp[j]);
		}
		
       	fprintf(fp3, "\n");	
	}
	
	fclose(fp1); fclose(fp3); fp1= NULL; fp3 = NULL;
	// -------------------------------------------------------------------------
	//  Release space
	// -------------------------------------------------------------------------
	delete[] a_points; a_points = NULL;
	delete[] med; med = NULL;
	delete[] data; data = NULL;
	delete[] Inval; Inval = NULL;
	delete[] upp; upp = NULL;
	delete[] low; low = NULL;
	delete[] ref_p; ref_p = NULL;
	
	for(int i = 0; i < n_pts_; i++){
		delete[] proj[i];
	}
	delete[] proj; proj = NULL;
	return 0;						// success to return
}

int QALSH::write_para_file(			// write "para" file from disk
	char* fname)						// file name of "para" file
{
	FILE* fp = NULL;
	fp = fopen(fname, "r");
	if (fp) {						// ensure the file not exist
		error("QALSH: hash tables exist.\n", true);
	}

	fp = fopen(fname, "w");			// open "para" file to write
	if (!fp) {
		printf("I could not create %s.\n", fname);
		printf("Perhaps no such folder %s?\n", index_path_);
		return 1;					// fail to return
	}

	fprintf(fp, "n = %d\n", n_pts_);// write <n_pts_>
	fprintf(fp, "d = %d\n", dim_);	// write <dim_>
	fprintf(fp, "B = %d\n", B_);	// write <B_>
									// write <appr_ratio_>
	fprintf(fp, "ratio = %f\n", appr_ratio_);
	
	//fprintf(fp, "ref = %d\n", n_ref_);
	fprintf(fp, "ring = %d\n", n_ring_);
	fprintf(fp, "sd = %d\n", s_dim_);
	
	
	fprintf(fp, "w = %f\n", w_);	// write <w_>
	fprintf(fp, "p1 = %f\n", p1_);	// write <p1_>
	fprintf(fp, "p2 = %f\n", p2_);	// write <p2_>
									// write <alpha_>
	fprintf(fp, "alpha = %f\n", alpha_);
									// write <beta_>
	fprintf(fp, "beta = %f\n", beta_);
									// write <delta_>
	fprintf(fp, "delta = %f\n", delta_);

	fprintf(fp, "m = %d\n", m_);	// write <m_>
	fprintf(fp, "l = %d\n", l_);	// write <l_>

	int count = 0;
	for (int i = 0; i < m_; i++) {	// write <a_array_>
	    for(int j = 0; j < s_dim_; j++){
		    for (int s = 0; s < dim_; s++) {
			    fprintf(fp, "%f ", a_array_[count++]);
		    }
		    fprintf(fp, "\n");
		}
	}
	count = 0;
	for (int i = 0; i < m_; i++) {	// write <s_array_>
	    for(int j = 0; j < s_dim_; j++){
			fprintf(fp, "%f ", s_array_[count++]);
		}
		fprintf(fp, "\n");
	}
	if (fp) fclose(fp);				// close para file
	
	return 0;						// success to return
}

float QALSH::calc_hash_value(		// calc hash value
	int table_id,						// hash table id
	float* point)						// a point
{
	float ret = 0.0f;
	for (int i = 0; i < dim_; i++) {
		ret += (a_array_[table_id * dim_ + i] * point[i]);  
	}
	return ret;
}

void QALSH::get_tree_filename(		// get file name of b-tree      
	int tree_id,						// tree id, from 0 to m-1
	char* fname)						// file name (return)
{
	char c[20];

	strcpy(fname, index_path_);
	sprintf(c, "%d", tree_id);
	strcat(fname, c);
	strcat(fname, ".qalsh");
}

// -----------------------------------------------------------------------------
int QALSH::restore(					// load existing b-trees.
	char* output_folder,
	int ring)				// folder of info of qalsh
{
									// init <index_path_>
	strcpy(index_path_, output_folder);
	strcat(index_path_, "L2_indices/");

	char fname[200];
	strcpy(fname, index_path_);
	strcat(fname, "para");

	if (read_para_file(fname)) {	// read "para" file and init params
		return 1;					// fail to return
	}

	trees_ = new BTree*[m_];		// allocate <trees>
	for (int i = 0; i < m_; i++) {
		get_tree_filename(i, fname);// get filename of tree

		trees_[i] = new BTree();	// init <trees>
		trees_[i]->init_restore(fname, ring);
	}
	return 0;						// success to return
}

// -----------------------------------------------------------------------------
int QALSH::read_para_file(			// read "para" file
	char* fname)						// file name of "para" file
{
	FILE* fp = NULL;
	fp = fopen(fname, "r");
	if (!fp) {						// ensure we can open the file
		printf("QALSH::read_para_file could not open %s.\n", fname);
		return 1;
	}

	fscanf(fp, "n = %d\n", &n_pts_);// read <n_pts_>
	fscanf(fp, "d = %d\n", &dim_);	// read <dim_>
	fscanf(fp, "B = %d\n", &B_);	// read <B_>
									// read <appr_ratio_>
	fscanf(fp, "ratio = %f\n", &appr_ratio_);
	
	//fscanf(fp, "ref = %d\n", &n_ref_);
	fscanf(fp, "ring = %d\n", &n_ring_);
	fscanf(fp, "sd = %d\n", &s_dim_);
	
	fscanf(fp, "w = %f\n", &w_);	// read <w_>
	fscanf(fp, "p1 = %f\n", &p1_);	// read <p1_>
	fscanf(fp, "p2 = %f\n", &p2_);	// read <p2_>
									// read <alpha_>
	fscanf(fp, "alpha = %f\n", &alpha_);
									// read <beta_>
	fscanf(fp, "beta = %f\n", &beta_);
									// read <delta_>
	fscanf(fp, "delta = %f\n", &delta_);

	fscanf(fp, "m = %d\n", &m_);	// read <m_>
	fscanf(fp, "l = %d\n", &l_);	// read <l_>

	a_array_ = new float[m_ * s_dim_ * dim_];// read <a_array_>
	s_array_ = new float[m_ * s_dim_];

	int count = 0;
	for(int i = 0; i < m_; i++){
		for(int j = 0; j < s_dim_; j++){
			for(int s = 0; s < dim_; s++){
	            fscanf(fp, "%f ", &a_array_[count]);
	            count++;
			}
	    fscanf(fp, "\n");
	    }
    }
	count = 0;
	for(int i = 0; i < m_; i++){
		for(int j = 0; j < s_dim_; j++){
	        fscanf(fp, "%f ", &s_array_[count]);
	        count++;
	    }
		fscanf(fp, "\n");
    }
	
	if (fp) fclose(fp);				// close para file
	
//	display_params();				// display params
	return 0;						// success to return
}

// -----------------------------------------------------------------------------
int QALSH::knn(						// k-nn search
	float* query,						// query point
	int top_k,							// top-k value
	float ratio,
	ResultItem* rslt,					// k-nn results
	char* output_folder,				// output folder
	float** center,
	float** r_lo,
	float** r_up,
	float** r_lo_sqr,
	float** r_up_sqr,
	int** Root_I,
	int* last_p,
	int** low,
	int** upp,
	float lam)
	
{
    char dName[100];
    strcpy(dName,output_folder);
    strcat(dName, "data"); 
    BlockFile2* d_blockFile = new BlockFile2(dName, 4096);
	
	boost::math::chi_squared chi(s_dim_);
	float prob = lam;
	float P_ = 0.9;
	float lambda = boost::math::quantile(chi, prob); 
	int thres = int( m_ * (prob - sqrt( -1 * log(1-P_) / 2 / m_ ) ) );
		
	// -------------------------------------------------------------------------
	//  Space allocation and initialization
	// -------------------------------------------------------------------------
									// init k-nn results
	for (int i = 0; i < top_k; i++) {
		rslt[i].id_   = -1;
		rslt[i].dist_ = MAXREAL;
	}
									// objects frequency
	int* frequency  = new int[n_pts_];
	for (int i = 0; i < n_pts_; i++) {
		frequency[i]  = 0;
	}
									// whether an object is checked
	bool* is_checked = new bool[n_pts_];
	for (int i = 0; i < n_pts_; i++) {
		is_checked[i] = false;
	}

	float* data = new float[dim_];	 // one object data
	for (int i = 0; i < dim_; i++) { 
		data[i] = 0.0f;
	}
	
	bool** fflag = new bool*[m_];

	for(int i = 0; i < m_; i++){
		fflag[i] = new bool[n_ring_];		
	}
	for(int i = 0; i < m_; i++){
		for(int j = 0; j < n_ring_; j++){
		    fflag[i][j] = false;
		}			
	}
	
    float sum = 0;      
    float** q_proj = new float*[m_]; 	
	for(int i = 0; i < m_; i++)
		q_proj[i] = new float[s_dim_];
	
	for(int i = 0; i < m_; i++){
		for(int j = 0; j <s_dim_; j++){
		    sum = 0;
		    for(int s = 0; s < dim_; s++){
		        sum += query[s]*a_array_[i*s_dim_*dim_+j*dim_+s];
		    }
			q_proj[i][j] = sum;	
		}
	}

	float* q_val = new float[m_];	
	float* q_val_sqr = new float[m_];

		
	
	for (int i = 0; i < m_; i++) {
		q_val[i] = -1.0f;
	}

	PageBuffer** lptr = new PageBuffer*[m_];

    for(int i = 0; i < m_; i++)	
	    lptr[i] = new PageBuffer[n_ring_];
	
	PageBuffer** rptr = new PageBuffer*[m_];
	
    for(int i = 0; i < m_; i++)	
	    rptr[i] = new PageBuffer[n_ring_];

	float* dis_ = new float[m_];
	float* dis_sqr = new float[m_];

	for (int i = 0; i < m_; i++) {  
		for(int k = 0; k < n_ring_; k++){
				
		    lptr[i][k].leaf_node_ = NULL;
		    lptr[i][k].index_pos_ = -1;
		    lptr[i][k].leaf_pos_  = -1;
		    lptr[i][k].size_      = -1;

		    rptr[i][k].leaf_node_ = NULL;
		    rptr[i][k].index_pos_ = -1;
		    rptr[i][k].leaf_pos_  = -1;
		    rptr[i][k].size_      = -1;
		}	
	}
		
    float sum2=0;	
	for(int i = 0; i < m_; i++){  
		sum = 0;
		sum2 = 0;	
	    for(int j = 0; j < s_dim_; j++){
			sum2 += (q_proj[i][j] - center[i][j]) * (q_proj[i][j] - center[i][j]);
			sum += s_array_[i * s_dim_ + j] * (q_proj[i][j] - center[i][j]);
		}
		q_val[i] = sum / sqrt(sum2);
			
	    q_val_sqr[i] = sqrt(1 - q_val[i]*q_val[i]);
		sum = 0;
	    for(int j = 0; j < s_dim_; j++){
			if(j == 0)
				{sum += (q_proj[i][j] - center[i][j]) / s_array_[i * s_dim_ + j];}
		    else{sum += (-1) * (q_proj[i][j] - center[i][j]) / s_array_[i * s_dim_ + j];}
		}
	    if(sum < 0){
		    q_val[i] = 2 - q_val[i];
		}
	    
	}
	
	for(int i = 0; i < m_; i++){
	    dis_sqr[i] = 0;
		for(int s = 0; s < s_dim_; s++){
			dis_sqr[i] += (center[i][s] - q_proj[i][s]) * (center[i][s] - q_proj[i][s]);
		}
		dis_[i] = sqrt(dis_sqr[i]);	
	}
	// -------------------------------------------------------------------------
	//  Compute hash value <q_dist> of query and init the page buffers 
	//  <lptr> and <rptr>.
	// -------------------------------------------------------------------------
	page_io_ = 0;					// num of page i/os
	dist_io_ = 0;					// num of dist cmpt
	
	float radius = -1;
	bool again      = true;			// stop flag
	int  candidates = 99 + top_k;	// threshold of candidates
	int  flag_num   = 0;			// used for bucket bound
	int  scanned_id = 0;			// num of scanned id

	int id    = -1;					// current object id
	int count = -1;					// count size in one page
	int start = -1;					// start position
	int end   = -1;					// end position

	float left_dist = -1.0f;		// left dist with query
	float right_dist = -1.0f;		// right dist with query
	float knn_dist = MAXREAL;		// kth nn dist
									// result entry for update
	ResultItem* item = new ResultItem();
	g_memory += (long) sizeof(ResultItem);
	
	vector<Elem> min_heap; 
	
    min_heap.reserve(2 * m_ * n_ring_);
    float temp= 0;
    int count2 = 0;
	
	Elem element;
	Elem* min_d = new Elem[m_ * n_ring_];
	
	for (int i = 0; i < m_; i++) {           
		for(int s = 0; s < n_ring_; s++){
            if(r_lo[i][s] < -0.5) continue; 
				
		    min_d[count2].tree = i;
			min_d[count2].ring = s;
				
			if(dis_[i] < r_lo[i][s]){
				min_d[count2].val = r_lo[i][s] - dis_[i];
				count2++;
				continue;	
			}	
			if(dis_[i] > r_up[i][s]){
				min_d[count2].val = dis_[i] - r_up[i][s];
				count2++;
			}
		}	
    }	

    qsort(min_d, count2, sizeof(Elem), ElemQsortComp);	
	
	float** cos_lo = new float* [m_];
    float** cos_up = new float* [m_];    

    for(int i = 0; i < m_; i++){
		cos_lo[i] = new float[n_ring_];
		cos_up[i] = new float[n_ring_];	
	}
    for(int i = 0; i < m_; i++){
		for(int s = 0; s < n_ring_; s++){
			cos_lo[i][s] = -2;
			cos_up[i][s] = -2;
	    }	
	}

	for (int i = 0; i < m_; i++){           
		for(int s = 0; s < n_ring_; s++){
			if(r_lo[i][s] < -0.5) continue;
			if(r_lo[i][s] <= dis_[i])     
				cos_lo[i][s] = r_lo[i][s]/dis_[i];
			if(r_up[i][s] <= dis_[i])	
				cos_up[i][s] = r_up[i][s]/dis_[i];
		}	
	}	
	
	float value;   
	PageBuffer* ptr;
	float key;
	int pos;
	
	float key0;
	float q_val0;
	for (int i = 0; i < m_; i++) {           
	    for(int k = 0; k < n_ring_; k++){				
		    if(dis_[i] <= r_up[i][k] && dis_[i] >= r_lo[i][k]){
				
				init_buffer(&(lptr[i][k]), &(rptr[i][k]), q_val[i], i, k, Root_I, last_p, fflag, low, upp);
					
                if(fflag[i][k] == false){  
				    ptr= &(lptr[i][k]);
				    pos = ptr->index_pos_;
	                key = ptr->leaf_node_->get_key(pos);
					if(key <= 1 && q_val[i] <= 1){
						temp = key* q_val[i] + ( sqrt(1- key*key) * q_val_sqr[i] );
						if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[i];}
				    }
					else if(key > 1 && q_val[i] <= 1){
						key0 = 2 - key;
						temp = key0 * q_val[i] - ( sqrt(1- key0 * key0) * q_val_sqr[i] );
						if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[i];}
					}
					else if(key <= 1 && q_val[i] > 1){
						q_val0 = 2 - q_val[i];
						temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[i]);
						if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
					}
					else{
						key0 = 2 - key;
						q_val0 = 2 - q_val[i];
						temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[i]);
						if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
					}
				    if(temp < cos_lo[i][k])
					    value = r_lo_sqr[i][k] + dis_sqr[i] - (2 * r_lo[i][k]* dis_[i] * temp); 
                    else{value = dis_sqr[i] * (1 - temp*temp);}    					
				    
			        element.tree = i;
				   // element.ref = j;
				    element.ring = k;
					element.dir = true;
			        element.val = value;
				    min_heap.push_back(element);   
				}
					
				ptr= &(rptr[i][k]);
				pos = ptr->index_pos_;
	                        key = ptr->leaf_node_->get_key(pos);
				if(key <= 1 && q_val[i] <= 1){
				    temp = key* q_val[i] + ( sqrt(1- key*key) * q_val_sqr[i] );
				    if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[i];}
				}
				else if(key > 1 && q_val[i] <= 1){
					key0 = 2 - key;
					temp = key0 * q_val[i] - ( sqrt(1- key0 * key0) * q_val_sqr[i]);
					if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[i];}
				}
				else if(key <= 1 && q_val[i] > 1){
					q_val0 = 2 - q_val[i];
					temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[i]);
					if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
				}
				else{
					key0 = 2 - key;
					q_val0 = 2 - q_val[i];
					temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[i]);
					if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
				}
					
				if(temp < cos_lo[i][k])
					value = r_lo_sqr[i][k] + dis_sqr[i] - (2 * r_lo[i][k]* dis_[i] * temp); 
                else{value = dis_sqr[i] * (1 - temp*temp);}    				
				    
			    element.tree = i;
				//element.ref = j;
				element.ring = k;
				element.dir = false;
			    element.val = value;
				min_heap.push_back(element);   
					
			}
		}
        
    }
	if(min_heap.empty() == false){
	    make_heap(min_heap.begin(),min_heap.end(), ElemHeapComp);  
	} 
	
	int ind = 0;
	int ii,ss;
	Elem tmp_elem;
	while(radius <= 0 ||  knn_dist >= ratio * sqrt( radius / lambda) ){             
	    if(min_heap.empty() == true){
	      if(ind < count2) {goto label;}
			else { break;} 
		}
		pop_heap(min_heap.begin(),min_heap.end(), ElemHeapComp);
		tmp_elem = min_heap[min_heap.size()-1];

		min_heap.pop_back();
		ii = tmp_elem.tree;
		ss = tmp_elem.ring;
		radius = tmp_elem.val;

		if(tmp_elem.dir == true){        
            count = lptr[ii][ss].size_;
			end = lptr[ii][ss].leaf_pos_;
			start = end - count;
			for (int j = end; j > start; j--) {
				id = lptr[ii][ss].leaf_node_->get_entry_id(j);
				frequency[id]++;
				
				if (frequency[id] > thres && !is_checked[id]) {
					is_checked[id] = true;
					read_data2(id, dim_, data, d_blockFile);
					item->dist_ = calc_l2_dist(data, query, dim_);
					item->id_ = id;
					knn_dist = update_result(rslt, item, top_k);
					dist_io_++;
				}
			}
			if(fflag[ii][ss] == true) continue;
		    update_left_buffer(&(lptr[ii][ss]), &(rptr[ii][ss]), fflag, low, upp, ii, ss);
		    if (fflag[ii][ss] == false) {          
				ptr= &(lptr[ii][ss]);
				pos = ptr->index_pos_;
	            key = ptr->leaf_node_->get_key(pos);
				if(key <= 1 && q_val[ii] <= 1){
					temp = key* q_val[ii] + ( sqrt(1- key*key) * q_val_sqr[ii] );
					if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[ii];}
				}
				else if(key > 1 && q_val[ii] <= 1){
					key0 = 2 - key;
					temp = key0 * q_val[ii] - ( sqrt(1- key0 * key0) * q_val_sqr[ii]);
					if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[ii];}
				}
				else if(key <= 1 && q_val[ii] > 1){
					q_val0 = 2 - q_val[ii];
					temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[ii]);
					if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
				}
				else{
					key0 = 2 - key;
					q_val0 = 2 - q_val[ii];
					temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[ii]);
					if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
				}			

				if(cos_lo[ii][ss] < -1.5 && cos_up[ii][ss] < -1.5)
					value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
				if(cos_lo[ii][ss] > -1.5 && cos_up[ii][ss] < -1.5){
					if(temp < cos_lo[ii][ss])
					    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp); 
                    else{value = dis_sqr[ii] * (1 - temp*temp);}  						
	            }
				if(cos_up[ii][ss] > -1.5){
					if(temp > cos_up[ii][ss])
						value = r_up_sqr[ii][ss] + dis_sqr[ii] - (2 * r_up[ii][ss]* dis_[ii] * temp);
					else if(temp < cos_lo[ii][ss])
						value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
					else{value = dis_sqr[ii] * (1 - temp*temp);}
				}
					
			    element.tree = ii;
				element.ring = ss;
			    element.dir = true;
			    element.val = value;
				min_heap.push_back(element);   
			    push_heap(min_heap.begin(), min_heap.end(), ElemHeapComp);
			}    
	    }
		else{
            count = rptr[ii][ss].size_;
			start = rptr[ii][ss].leaf_pos_;
			end = start + count;
			for (int j = start; j < end; j++) {
				id = rptr[ii][ss].leaf_node_->get_entry_id(j);
				frequency[id]++;
				
				if (frequency[id] > thres && !is_checked[id]) {
					is_checked[id] = true;
					read_data2(id, dim_, data, d_blockFile);

					item->dist_ = calc_l2_dist(data, query, dim_);
					item->id_ = id;
					knn_dist = update_result(rslt, item, top_k);
					dist_io_++;
				}
			}
			if(fflag[ii][ss] == true) continue;
			
		    update_right_buffer(&(lptr[ii][ss]), &(rptr[ii][ss]), fflag, low, upp, ii, ss);
		    if (fflag[ii][ss] == false) {          
				ptr= &(rptr[ii][ss]);
				pos = ptr->index_pos_;
	            key = ptr->leaf_node_->get_key(pos);
				
				if(key <= 1 && q_val[ii] <= 1){
					temp = key* q_val[ii] + ( sqrt(1- key*key) * q_val_sqr[ii] );
					if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[ii];}
				}
				else if(key > 1 && q_val[ii] <= 1){
					key0 = 2 - key;
					temp = key0 * q_val[ii] - ( sqrt(1- key0 * key0) * q_val_sqr[ii] );
					if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[ii];}
				}
				else if(key <= 1 && q_val[ii] > 1){
					q_val0 = 2 - q_val[ii];
					temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[ii]);
					if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
				}
				else{
					key0 = 2 - key;
					q_val0 = 2 - q_val[ii];
					temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[ii]);
					if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
				}

				if(cos_lo[ii][ss] < -1.5 && cos_up[ii][ss] < -1.5)
					value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
				if(cos_lo[ii][ss] > -1.5 && cos_up[ii][ss] < -1.5){
					if(temp < cos_lo[ii][ss])
					    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp); //切点下移
                    else{value = dis_sqr[ii] * (1 - temp*temp);}  						
	            }
				if(cos_up[ii][ss] > -1.5){
					if(temp > cos_up[ii][ss])
						value = r_up_sqr[ii][ss] + dis_sqr[ii] - (2 * r_up[ii][ss]* dis_[ii] * temp);
					else if(temp < cos_lo[ii][ss])
						value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
					else{value = dis_sqr[ii] * (1 - temp*temp);}
				}				
				    
			    element.tree = ii;
				element.ring = ss;
			    element.dir = false;
			    element.val = value;
				min_heap.push_back(element);   
			    push_heap(min_heap.begin(), min_heap.end(), ElemHeapComp);
			}						
		}

label:	for(; ind < count2; ind++){
			if(min_heap.empty() == true || radius >= min_d[ind].val * min_d[ind].val){
				ii = min_d[ind].tree;
				ss = min_d[ind].ring;

				init_buffer(&(lptr[ii][ss]), &(rptr[ii][ss]), q_val[ii], ii, ss, Root_I, last_p, fflag, low, upp);

			    if(fflag[ii][ss] == false){   
				    ptr= &(lptr[ii][ss]);
				    pos = ptr->index_pos_;
	                key = ptr->leaf_node_->get_key(pos);
				    if(key <= 1 && q_val[ii] <= 1){
					    temp = key* q_val[ii] + ( sqrt(1- key*key) * q_val_sqr[ii] );
					    if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[ii];}
				    }
				    else if(key > 1 && q_val[ii] <= 1){
					    key0 = 2 - key;
					    temp = key0 * q_val[ii] - ( sqrt(1- key0 * key0) * q_val_sqr[ii] );
					    if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[ii];}
				    }
				    else if(key <= 1 && q_val[ii] > 1){
					    q_val0 = 2 - q_val[ii];
					    temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[ii]);
					    if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
				    }
				    else{
					    key0 = 2 - key;
					    q_val0 = 2 - q_val[ii];
					    temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[ii]);
					    if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
				    }
				    if(cos_lo[ii][ss] < -1.5 && cos_up[ii][ss] < -1.5)
					    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
				    if(cos_lo[ii][ss] > -1.5 && cos_up[ii][ss] < -1.5){
					    if(temp < cos_lo[ii][ss])
					        value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp); //切点下移
                        else{value = dis_sqr[ii] * (1 - temp*temp);}  						
	                }
				    if(cos_up[ii][ss] > -1.5){
					    if(temp > cos_up[ii][ss])
						    value = r_up_sqr[ii][ss] + dis_sqr[ii] - (2 * r_up[ii][ss]* dis_[ii] * temp);
					    else if(temp < cos_lo[ii][ss])
						    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
					    else{value = dis_sqr[ii] * (1 - temp*temp);}
				    }   				
				    
			        element.tree = ii;
				    element.ring = ss;
			        element.dir = true;
			        element.val = value;
				    min_heap.push_back(element);  
			        push_heap(min_heap.begin(), min_heap.end(), ElemHeapComp);
				}
				
				    ptr= &(rptr[ii][ss]);
				    pos = ptr->index_pos_;
	                key = ptr->leaf_node_->get_key(pos);
				    if(key <= 1 && q_val[ii] <= 1){
					    temp = key* q_val[ii] + ( sqrt(1- key*key) * q_val_sqr[ii] );
					    if(!(temp >= -1 && temp <= 1)) {temp = key* q_val[ii];}
				    }
				    else if(key > 1 && q_val[ii] <= 1){
					    key0 = 2 - key;
					    temp = key0 * q_val[ii] - ( sqrt(1- key0 * key0) * q_val_sqr[ii] );
					    if(!(temp >= -1 && temp <= 1)) {temp = key0* q_val[ii];}
				    }
				    else if(key <= 1 && q_val[ii] > 1){
					    q_val0 = 2 - q_val[ii];
					    temp = key* q_val0 - (sqrt(1- key*key) * q_val_sqr[ii]);
					    if(!(temp >= -1 && temp <= 1)) {temp = key* q_val0;}
				    }
				    else{
					    key0 = 2 - key;
					    q_val0 = 2 - q_val[ii];
					    temp = key0 * q_val0 + (sqrt(1- key0 * key0) * q_val_sqr[ii]);
					    if(!(temp >= -1 && temp <= 1)) {temp = key0 * q_val0;}
				    }
				    if(cos_lo[ii][ss] < -1.5 && cos_up[ii][ss] < -1.5)
					    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
				    if(cos_lo[ii][ss] > -1.5 && cos_up[ii][ss] < -1.5){
					    if(temp < cos_lo[ii][ss])
					        value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp); //切点下移
                        else{value = dis_sqr[ii] * (1 - temp*temp);}  						
	                }
				    if(cos_up[ii][ss] > -1.5){
					    if(temp > cos_up[ii][ss])
						    value = r_up_sqr[ii][ss] + dis_sqr[ii] - (2 * r_up[ii][ss]* dis_[ii] * temp);
					    else if(temp < cos_lo[ii][ss])
						    value = r_lo_sqr[ii][ss] + dis_sqr[ii] - (2 * r_lo[ii][ss]* dis_[ii] * temp);
					    else{value = dis_sqr[ii] * (1 - temp*temp);}
				    }	
				    
			        element.tree = ii;
				    element.ring = ss;
			        element.dir = false;
			        element.val = value;
				    min_heap.push_back(element);   
			        push_heap(min_heap.begin(), min_heap.end(), ElemHeapComp);  
			}
			else { break;}			
		}
		if(min_heap.size() == 0) break;
	}
	
	for(int i = 0; i < m_; i++){
		for(int k = 0; k < n_ring_; k++){
			if (lptr[i][k].leaf_node_ && lptr[i][k].leaf_node_ != rptr[i][k].leaf_node_) {
			    delete lptr[i][k].leaf_node_; lptr[i][k].leaf_node_ = NULL;
		    }
		    if (rptr[i][k].leaf_node_) {
			    delete rptr[i][k].leaf_node_; rptr[i][k].leaf_node_ = NULL;
		    } 
		}
	}
	
	for(int i = 0; i < m_; i++){
		delete[] lptr[i];
		delete[] rptr[i];
		delete[] cos_lo[i];
		delete[] cos_up[i];
		delete[] q_proj[i];
		delete[] fflag[i];
	}
	delete[] lptr; lptr = NULL;
	delete[] rptr; rptr = NULL;
	delete[] q_val; q_val = NULL;
	delete[] q_val_sqr; q_val_sqr = NULL;
	delete[] cos_lo; cos_lo = NULL;
	delete[] cos_up; cos_up = NULL;
	delete[] min_d; min_d = NULL;
	delete[] dis_sqr; dis_sqr = NULL;
	delete[] dis_; dis_ = NULL;
	delete[] q_proj; q_proj = NULL;	
	delete[] fflag; fflag = NULL;
	
	
	
	if (data != NULL || frequency != NULL || is_checked != NULL) {
		delete[] data; data = NULL;
		delete[] frequency;  frequency  = NULL;
		delete[] is_checked; is_checked = NULL;
		g_memory -= ((SIZEBOOL + SIZEINT) * n_pts_ + SIZEFLOAT * dim_);
	}
	if (item != NULL) {
		delete item; item = NULL;
		g_memory -= (SIZEFLOAT + SIZEBOOL) * m_;
		g_memory -= (long) sizeof(ResultItem);
	}
	if (d_blockFile){
        delete d_blockFile;
    }
	return (page_io_ + dist_io_);
//	return (dist_io_);
}


void QALSH::init_buffer(			// init page buffer (loc pos of b-treee)
	PageBuffer* lptr,					// left buffer page (return)
	PageBuffer* rptr,					// right buffer page (return)
	float q_val,						// hash value of query (return)
	int i,
	int k,
	int** Root_I,
	int* last_p,
	bool** fflag,
	int** low,
	int** upp)
{ 
	int  block   = -1;				// tmp vars for index node
	int  follow  = -1;
	bool lescape = false;

	int pos = -1;					// tmp vars for leaf node
	int increment = -1;
	int num_entries = -1;

	BIndexNode* index_node = NULL;
	block = Root_I[i][k];
	
	if(block <= last_p[i]){ //root node is also leaf node
		BLeafNode* leaf_node = new BLeafNode();
	    leaf_node->init_restore(trees_[i], block);
		page_io_++;
		(*rptr).leaf_node_ = leaf_node;
		(*rptr).block_ = block;
		(*lptr).leaf_node_ = leaf_node;
		(*lptr).block_ = block;
		
		
		increment = (*rptr).leaf_node_->get_increment();
		
		if(leaf_node->get_num_keys() <= 1){ //only one key
			(*rptr).index_pos_ = 0;
		    (*rptr).leaf_pos_ = 0;
			(*rptr).size_ = num_entries;

			fflag[i][k] = true;
		}
		else{
		    pos = leaf_node->find_position_by_key(q_val);
		    if (pos < 0 || pos >= leaf_node->get_num_keys() - 1){
				rptr->index_pos_ = 0;
			    rptr->leaf_pos_ = 0;
				rptr->size_ = increment;
				
				lptr->index_pos_ = leaf_node->get_num_keys() - 1;
				pos = lptr->index_pos_;
				lptr->leaf_pos_ = num_entries - 1;
			    lptr->size_ = num_entries - pos * increment;
			}
		    else{
			    rptr->index_pos_ = (pos + 1);
			    rptr->leaf_pos_ = (pos + 1) * increment;
			    if ((pos + 1) == (*rptr).leaf_node_->get_num_keys() - 1) {
				    num_entries = (*rptr).leaf_node_->get_num_entries();
				    (*rptr).size_ = num_entries - (pos + 1) * increment;
			    }
			    else {
				    (*rptr).size_ = increment;
			    }
				
				(*lptr).index_pos_ = pos;
				(*lptr).leaf_pos_ = pos * increment + increment - 1;
			    (*lptr).size_ = increment;
			}
		}
	} 
	else{
	index_node = new BIndexNode();
	index_node->init_restore(trees_[i], block);
	page_io_++;

		// ---------------------------------------------------------------------
		//  Find the leaf node whose value is closest and larger than the key
		//  of query q <qe->key>
		// ---------------------------------------------------------------------
	lescape = false;			// locate the position of branch	
	while (index_node->get_level() > 1) {
		follow = index_node->find_position_by_key(q_val);

		if (follow == -1) {		// if in the most left branch
			if (lescape) {		// scan the most left branch
				follow = 0;
			}
			else {
				if (block != Root_I[i][k]) {
					error("QALSH::knn_bucket No branch found\n", true);
				}
				else {
					follow = 0;
					lescape = true;
				}
			}
		}
		block = index_node->get_son(follow);
		delete index_node; index_node = NULL;

		index_node = new BIndexNode();
		index_node->init_restore(trees_[i], block);
		page_io_++;				// access a new node (a new page)
	}

		// ---------------------------------------------------------------------
		//  After finding the leaf node whose value is closest to the key of
		//  query, initialize <lptrs[i]> and <rptrs[i]>.
		//
		//  <lescape> = true is that the query has no <lptrs>, the query is 
		//  the smallest value.
		// ---------------------------------------------------------------------
	follow = index_node->find_position_by_key(q_val);
	if (follow < 0) {
		lescape = true;
		follow = 0;
	}
 
	if (lescape) {				// 
		block = index_node->get_son(0);
		(*rptr).leaf_node_ = new BLeafNode();
		(*rptr).leaf_node_->init_restore(trees_[i], block);
		(*rptr).index_pos_ = 0;
		(*rptr).leaf_pos_ = 0;
        (*rptr).block_ = block;
		
		increment = (*rptr).leaf_node_->get_increment();
		num_entries = (*rptr).leaf_node_->get_num_entries();
		if (increment > num_entries) {
			(*rptr).size_ = num_entries;
		} 
		else{
			(*rptr).size_ = increment;
		} 
		page_io_++;
	    block = upp[i][k];
		(*lptr).block_ = block;
		(*lptr).leaf_node_ = new BLeafNode();
		(*lptr).leaf_node_->init_restore(trees_[i], block);
        lptr->index_pos_ = lptr->leaf_node_->get_num_keys() - 1;
			
	    pos = lptr->index_pos_;
		increment = (*lptr).leaf_node_->get_increment();
	    num_entries = (*lptr).leaf_node_->get_num_entries();
		(*lptr).leaf_pos_ = num_entries - 1;
		(*lptr).size_ = num_entries - pos * increment;

		page_io_++;				
	}
	else {						// init left buffer
		block = index_node->get_son(follow);
		(*lptr).block_ = block;
		(*lptr).leaf_node_ = new BLeafNode();
		(*lptr).leaf_node_->init_restore(trees_[i], block);

		pos = (*lptr).leaf_node_->find_position_by_key(q_val);
		if (pos < 0) pos = 0;
		(*lptr).index_pos_ = pos;

		increment = (*lptr).leaf_node_->get_increment();
		if (pos == (*lptr).leaf_node_->get_num_keys() - 1) {
			num_entries = (*lptr).leaf_node_->get_num_entries();
			(*lptr).leaf_pos_ = num_entries - 1;
			(*lptr).size_ = num_entries - pos * increment;
		}
		else {
			(*lptr).leaf_pos_ = pos * increment + increment - 1;
			(*lptr).size_ = increment;
		}
		page_io_++;
							// init right buffer
	    if (pos < (*lptr).leaf_node_->get_num_keys() - 1) {
			(*rptr).block_ = block;
			(*rptr).leaf_node_ = (*lptr).leaf_node_;
			(*rptr).index_pos_ = (pos + 1);
			(*rptr).leaf_pos_ = (pos + 1) * increment;
				
			if ((pos + 1) == (*rptr).leaf_node_->get_num_keys() - 1) {
				num_entries = (*rptr).leaf_node_->get_num_entries();
				(*rptr).size_ = num_entries - (pos + 1) * increment;
			}
			else {
				(*rptr).size_ = increment;
			}
		}
		else {
			if(block == upp[i][k]){                
			    block = low[i][k];
				(*rptr).block_ = block;
		        (*rptr).leaf_node_ = new BLeafNode();
		        (*rptr).leaf_node_->init_restore(trees_[i], block);
			
			    rptr->index_pos_ = 0;
			    rptr->leaf_pos_ = 0;

			    increment = rptr->leaf_node_->get_increment();
			    num_entries = rptr->leaf_node_->get_num_entries();
			    if (increment > num_entries) {
				    rptr->size_ = num_entries;
			    } else {
				    rptr->size_ = increment;
			    }	

		        page_io_++;		        		
			}
			else{
			    (*rptr).leaf_node_ = (*lptr).leaf_node_->get_right_sibling();
				(*rptr).index_pos_ = 0;
				(*rptr).leaf_pos_ = 0;
				(*rptr).block_ = block + 1;

				    increment = (*rptr).leaf_node_->get_increment();
				    num_entries = (*rptr).leaf_node_->get_num_entries();
				    if (increment > num_entries) {
					    (*rptr).size_ = num_entries;
				    } else {
					    (*rptr).size_ = increment;
				    }
				    page_io_++;
			    
			}
		}
	}
    
	if (index_node != NULL) {
		delete index_node; index_node = NULL;
	}
	}	
}

// -----------------------------------------------------------------------------
float QALSH::find_radius(			// find proper radius
	PageBuffer* lptr,					// left page buffer
	PageBuffer* rptr,					// right page buffer
	float* q_dist)						// hash value of query
{
	float radius = update_radius(lptr, rptr, q_dist, 1.0f/appr_ratio_);
	if (radius < 1.0f) radius = 1.0f;

	return radius;
}

// -----------------------------------------------------------------------------
float QALSH::update_radius(			// update radius
	PageBuffer* lptr,					// left page buffer
	PageBuffer* rptr,					// right page buffer
	float* q_dist,						// hash value of query
	float  old_radius)					// old radius
{
	float dist = 0.0f;				// tmp vars
	vector<float> list;

	for (int i = 0; i < m_; i++) {	// find an array of proj dist
		if (lptr[i].size_ != -1) {
			dist = calc_proj_dist(&lptr[i], q_dist[i]);
			list.push_back(dist);		
		}
		if (rptr[i].size_ != -1) {
			dist = calc_proj_dist(&rptr[i], q_dist[i]);
			list.push_back(dist);
		}
	}
	sort(list.begin(), list.end());	// sort the array

	int num = (int) list.size();
	if (num == 0) return appr_ratio_ * old_radius;
	
	if (num % 2 == 0) {				// find median dist
		dist = (list[num/2 - 1] + list[num/2]) / 2.0f;
	} else {
		dist = list[num/2];
	}
	list.clear();

	int kappa = (int) ceil(log(2.0f * dist / w_) / log(appr_ratio_));  //k of c^k in paper
	dist = pow(appr_ratio_, kappa);
	
	return dist;
}

// -----------------------------------------------------------------------------
float QALSH::update_result(			// update knn results
	ResultItem* rslt,					// k-nn results
	ResultItem* item,					// new result
	int top_k)							// top-k value
{
	int i = -1;
	int pos = -1;
	bool alreadyIn = false;

	for (i = 0; i < top_k; i++) {
									// ensure the id is not exist before
		if (item->id_ == rslt[i].id_) {
			alreadyIn = true;
			break;
		}							// find the position to insert
		else if (compfloats(item->dist_, rslt[i].dist_) == -1) {
			break;
		}
	}
	pos = i;

	if (!alreadyIn && pos < top_k) {// insertion
		for (i = top_k - 1; i > pos; i--) {
			rslt[i].setto(&(rslt[i - 1]));
		}
		rslt[pos].setto(item);
	}
	return rslt[top_k - 1].dist_;
}

// -----------------------------------------------------------------------------
void QALSH::update_left_buffer(		// update left buffer
	PageBuffer* lptr,					// left buffer
	const PageBuffer* rptr,				// right buffer
	bool** fflag,
	int** low,
	int** upp,
	int i,
	int k)
{
	BLeafNode* leaf_node = NULL;
	BLeafNode* old_leaf_node = NULL;
	int block;
	//printf("left_start\n");
	if (lptr->index_pos_ > 0) {	
		lptr->index_pos_--;
		
		if(lptr -> block_ == rptr -> block_ && lptr->index_pos_ == rptr->index_pos_)
            fflag[i][k] = true;
		else{
		    int pos = lptr->index_pos_;
		    int increment = lptr->leaf_node_->get_increment();
		    lptr->leaf_pos_ = pos * increment + increment - 1;
		    lptr->size_ = increment;
		}
	}
	else {
		old_leaf_node = lptr->leaf_node_;
		if(lptr -> block_ == low[i][k]){
			block = upp[i][k];
			if(rptr -> block_ == upp[i][k] && rptr->index_pos_ == rptr->leaf_node_->get_num_keys() - 1)
                fflag[i][k] = true;
            else{
				if(rptr -> block_ == upp[i][k])
				    leaf_node = rptr -> leaf_node_;
				else{ 
				    leaf_node = new BLeafNode();
				    leaf_node->init_restore(trees_[i], upp[i][k]);
					page_io_++;
				}
			}			
		}
		else{
			block = (lptr -> block_) - 1;
			if(lptr->block_ - rptr->block_ == 1 && rptr->index_pos_ == rptr->leaf_node_->get_num_keys() - 1)
			    fflag[i][k] = true;
			else{
				if(lptr->block_ - rptr->block_ == 1)
					leaf_node = rptr -> leaf_node_;
				else{
					leaf_node = lptr->leaf_node_->get_left_sibling();
				    page_io_++;
				}
			}
		}

		if(fflag[i][k] == false){
			lptr->block_ = block;
			lptr->leaf_node_ = leaf_node;
			lptr->index_pos_ = lptr->leaf_node_->get_num_keys() - 1;

			int pos = lptr->index_pos_;
			int increment = lptr->leaf_node_->get_increment();
			int num_entries = lptr->leaf_node_->get_num_entries();
			lptr->leaf_pos_ = num_entries - 1;
			lptr->size_ = num_entries - pos * increment;
	    }
		if (rptr->leaf_node_ != old_leaf_node) {
			delete old_leaf_node; old_leaf_node = NULL;
			if(fflag[i][k] == true) {lptr->leaf_node_ = NULL;}
		}
	}
}

// -----------------------------------------------------------------------------
void QALSH::update_right_buffer(	// update right buffer
	const PageBuffer* lptr,				// left buffer
	PageBuffer* rptr,					// right buffer
	bool** fflag,
	int** low,
	int** upp,
	int i,
	int k)
{
	BLeafNode* leaf_node = NULL;
	BLeafNode* old_leaf_node = NULL;
    int block;

	if (rptr->index_pos_ < rptr->leaf_node_->get_num_keys() - 1) {
		rptr->index_pos_++;

		if(lptr -> block_ == rptr -> block_ && lptr->index_pos_ == rptr->index_pos_)
			fflag[i][k] = true;
		else{
		    int pos = rptr->index_pos_;
		    int increment = rptr->leaf_node_->get_increment();
		    rptr->leaf_pos_ = pos * increment;
		    if (pos == rptr->leaf_node_->get_num_keys() - 1) {
			    int num_entries = rptr->leaf_node_->get_num_entries();
			    rptr->size_ = num_entries - pos * increment;
		    }
		    else {
			    rptr->size_ = increment;
		    }
	    }	
	}
	else {
		old_leaf_node = rptr->leaf_node_;
		if(rptr -> block_ == upp[i][k]){
			block = low[i][k];
			if(lptr -> block_ == low[i][k] && lptr->index_pos_ == 0)
                fflag[i][k] = true;
            else{
				if(lptr -> block_ == low[i][k])
				    leaf_node = lptr -> leaf_node_;
				else{ 
				    leaf_node = new BLeafNode();
				    leaf_node->init_restore(trees_[i], low[i][k]);
					page_io_++;
				}
			}			
		}
		else{
			block = (rptr->block_) + 1;
			if(lptr->block_ - rptr->block_ == 1 && lptr->index_pos_ == 0)
			    fflag[i][k] = true;
			else{
				if(lptr->block_ - rptr->block_ == 1)
					leaf_node = lptr -> leaf_node_;
				else{
					leaf_node = rptr->leaf_node_->get_right_sibling();
				    page_io_++;
				}
			}
		}

		if(fflag[i][k] == false){
		    rptr->block_ = block;
			rptr->leaf_node_ = leaf_node;
			rptr->index_pos_ = 0;
			rptr->leaf_pos_ = 0;

			int increment = rptr->leaf_node_->get_increment();
			int num_entries = rptr->leaf_node_->get_num_entries();
			if (increment > num_entries) {
				rptr->size_ = num_entries;
			} else {
				rptr->size_ = increment;
			}
		}

		if (lptr->leaf_node_ != old_leaf_node) {
			delete old_leaf_node; old_leaf_node = NULL;
			if(fflag[i][k] == true) {rptr->leaf_node_ = NULL;}
		}
	}
}

// -----------------------------------------------------------------------------
float QALSH::calc_proj_dist(		// calc proj dist
	const PageBuffer* ptr,				// page buffer
	float q_val)						// hash value of query
{
	int pos = ptr->index_pos_;
	float key = ptr->leaf_node_->get_key(pos);
	float dist = fabs(key - q_val);

	return dist;
}

// -----------------------------------------------------------------------------
//  Comparison function for qsort called by QALSH::bulkload()
// -----------------------------------------------------------------------------
int HashValueQsortComp(				// compare function for qsort
	const void* e1,						// 1st element
	const void* e2)						// 2nd element
{
	int ret = 0;
	HashValue* value1 = (HashValue *) e1;
	HashValue* value2 = (HashValue *) e2;

	if (value1->proj_ < value2->proj_) {
		ret = -1;
	} else if (value1->proj_ > value2->proj_) {
		ret = 1;
	} else {
		if (value1->id_ < value2->id_) ret = -1;
		else if (value1->id_ > value2->id_) ret = 1;
	}
	return ret;
}

int ElemQsortComp(				// compare function for qsort
	const void* e1,						// 1st element
	const void* e2)						// 2nd element
{
	int ret = 0;
	Elem* value1 = (Elem *) e1;
	Elem* value2 = (Elem *) e2;

	if (value1->val < value2->val) {
		ret = -1;
	} else{
		ret = 1;
	} 
	return ret;
}

bool ElemHeapComp(				// compare function for qsort
	const Elem& e1,						// 1st element
	const Elem& e2)
{
	if (e1.val < e2.val) {
		return false;
	} else{
		return true;
	} 
}
