#include "headers.h"

// -----------------------------------------------------------------------------
int ground_truth(					// output the ground truth results
	int   n,							// number of data points
	int   qn,							// number of query points
	int   d,							// dimension of space
	char* data_set,						// address of data set
	char* query_set,					// address of query set
	char* truth_set)					// address of ground truth file
{
	clock_t startTime = (clock_t) -1;  
	clock_t endTime   = (clock_t) -1;
	
	int maxk = MAXK;
	float dist = -1.0F;
	float* knndist = new float[maxk];
	int* recall = new int[maxk];
	float ind;

	FILE* fp = fopen(truth_set, "w");		// open output file
	FILE* fp0 = fopen("recall.txt", "w");
	
	if (!fp) {
		printf("I can not create %s.\n", truth_set);
		return 1;
	}

	fprintf(fp, "%d %d\n", qn, maxk);
	
	// -------------------------------------------------------------------------
	//  Read data set and query set
	// -------------------------------------------------------------------------
	
	int i, j, ii, jj;
    std::ifstream inD(data_set, std::ios::binary);	
    if (!inD.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
	
    std::ifstream inQ(query_set, std::ios::binary);	
    if (!inQ.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }	
	
        //char* data_byte = new char[d];			
	float* data = new float[d];		
	float* query = new float[d];
     
    inQ.seekg(0, std::ios::beg);	
	
	startTime = clock();
    for (i = 0; i < qn; i++) {
		inD.seekg(0, std::ios::beg);	
        inQ.seekg(4, std::ios::cur);	
        inQ.read((char*)(query), d * 4);	
		
		for (j = 0; j < maxk; j++) {
			knndist[j] = MAXREAL;
		}
		for (j = 0; j < n; j++) {
            inD.seekg(4, std::ios::cur);	
            inD.read((char*)(data), 4 * d);	
			
			dist = calc_l2_dist(data, query, d);
			for (jj = 0; jj < maxk; jj++) {
				if (compfloats(dist, knndist[jj]) == -1) {
					break;
				}
			}
			if (jj < maxk) {
				for (ii = maxk - 1; ii >= jj + 1; ii--) {
					knndist[ii] = knndist[ii - 1];
					recall[ii] = recall[ii - 1];
				}
				knndist[jj] = dist;
				recall[jj] = j+1;
			}			
		}
		fprintf(fp, "%d", i + 1);
		for (j = 0; j < maxk; j++) {
			fprintf(fp, " %f", knndist[j]);
		}
		fprintf(fp, "\n");
		
		for (j = 0; j < maxk; j++){
		    fprintf(fp0, "%d ", recall[j]);
		}
		fprintf(fp0, "\n");		
    }
	endTime = clock();
	printf("Generate Ground Truth: %.6f Seconds\n\n", 
	((float) endTime - (float) startTime) / CLOCKS_PER_SEC);
	inQ.close();
	inD.close();
	fclose(fp);						// close output file
	fclose(fp0);
	// -------------------------------------------------------------------------
	//  Release space
	// -------------------------------------------------------------------------
	if (data != NULL) {				// release <data>
		delete[] data; data = NULL;
	}
	if (query != NULL) {			// release <query>
        delete[] query; query = NULL;
	}
	if (knndist != NULL) {			// release <knndist>
		delete[] knndist; knndist = NULL;
	}
    if (recall != NULL) {
	    delete[] recall; recall = NULL;
	}	
	return 0;
}

int indexing(						// build hash tables for the dataset
	int   n,							// number of data points
	int   m,
	int   n_ring,
	int   d,							// dimension of space
	int   sub_d,                        // dimension of subspace
	int   B,							// page size
	char* data_set,						// address of data set
	char* output_folder)				// folder to store info of qalsh
{
	clock_t startTime = (clock_t) -1;
	clock_t endTime   = (clock_t) -1;
		
    char dName[100]; /* data file name (binary file) */
	
	int ret = 0;
    int cnt = -1;
    float son = -1;

    BlockFile2 * d_blockFile = NULL;
    char * blk = NULL;
    int blk_pos = -1;

    strcpy(dName,output_folder);
    strcat(dName, "data");
    
    d_blockFile = new BlockFile2(dName, 4096);
    blk = new char[d_blockFile->blocklength];
    blk_pos = 0;
    
    float* key = new float[d];
    cnt = 0;

	std::ifstream in(data_set, std::ios::binary); // open data file 
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
	in.seekg(0, std::ios::beg);

   //char* key0 = new char[d];	
    while ( cnt < n) {

        if(cnt % 10000000 == 0)
            printf("cnt = %d\n",cnt);
		
            in.seekg(4, std::ios::cur);	
            in.read((char*)(key), 4 * d);
 
            //for(int i2 = 0; i2 < d; i2 ++)
            //     key[i2] = key0[i2];	   
	    	
        for (int j = 0; j < d; ++j) {
            memcpy(&blk[blk_pos], &key[j], sizeof (float));
            blk_pos += sizeof (float);
            if (blk_pos == d_blockFile->blocklength) {
                blk_pos = 0;
                d_blockFile->append_block(blk);
            }
        }
       
        cnt++;
    }
    if (blk_pos != 0) {
        d_blockFile->append_block(blk); //Thus the last block is not clear!!!
    }
  
    delete d_blockFile;
    delete [] blk;
	in.close();	
			
	// -------------------------------------------------------------------------
	//  Bulkloading
	// -------------------------------------------------------------------------
	char fname[200];
	strcpy(fname, output_folder);
	strcat(fname, "L2_index.out");

	FILE* fp = fopen(fname, "w");
	if (!fp) {
		printf("I could not create %s.\n", fname);
		return 1;					// fail to return
	}

	startTime = clock();
	QALSH* lsh = new QALSH();
	lsh->init(n, m, n_ring, d, sub_d, B, output_folder);          
	lsh->bulkload(data_set);                                
	endTime = clock();

	float indexing_time = ((float) endTime - startTime) / CLOCKS_PER_SEC;
	printf("\nIndexing Time: %.6f seconds\n\n", indexing_time);
	fprintf(fp, "Indexing Time: %.6f seconds\n", indexing_time);
	fclose(fp);

	// -------------------------------------------------------------------------
	//  Release space
	// -------------------------------------------------------------------------

	if (lsh != NULL) {
		delete lsh; lsh = NULL;
	}
	return 0;
}

// -----------------------------------------------------------------------------
int lshknn(							// k-nn via qalsh (data in disk)
	int   qn,							// number of query points
	int   d,							// dimensionality
	char* query_set,					// path of query set
	char* truth_set,					// groundtrue file
	char* output_folder,				// output folder
	int   m,
	float ratio,
	int   n_ring,
	int   sub_d,
	float lam,
	int K,
	float c)
	
{
	int ret = 0;
	int maxk = MAXK;
	int i, j;
	FILE* fp = NULL;				// file pointer
	FILE* f1 =fopen("result.txt","a");
	// -------------------------------------------------------------------------
	//  Read query set
	// -------------------------------------------------------------------------
		
	float** query = new float*[qn];
        for(int i = 0; i < qn; i++) query[i] = new float[d];
	
	std::ifstream in(query_set, std::ios::binary);	
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
	in.seekg(0, std::ios::beg);
    for (i = 0; i < qn; i++) {
        in.seekg(4, std::ios::cur);	
        in.read((char*)(query[i]), d * 4);	
	}
	 
	// -------------------------------------------------------------------------
	//  Read the ground truth file
	// -------------------------------------------------------------------------
	g_memory += SIZEFLOAT * qn * maxk;
	float* R = new float[qn * maxk];
	int* recall = new int[qn * maxk];

	fp = fopen(truth_set, "r");		// open ground truth file
	if (!fp) {
		printf("Could not open the ground truth file.\n");
		return 1;
	}

	FILE* fp2 = fopen("recall.txt","r");
	if (!fp2){printf("error\n"); return 1;}
	
	fscanf(fp, "%d %d\n", &qn, &maxk);
	for (int i = 0; i < qn; i++) {
		fscanf(fp, "%d", &j);
		for (j = 0; j < maxk; j ++) {
			fscanf(fp, " %f", &(R[i * maxk + j]));
		}
	}
	
	for (int i = 0; i < qn; i++){
	    for(j = 0; j < maxk; j++){
	      fscanf(fp2, "%d ", &(recall[i * maxk + j]));
	    }
	}
	fclose(fp);						// close groundtrue file
	fclose(fp2);

	// -------------------------------------------------------------------------
	//  K-nearest neighbor (k-nn) search via qalsh
	// -------------------------------------------------------------------------
//	int kNNs[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
//	int maxRound = 11;
	int top_k = 0;

	float allTime   = -1.0f;
	float allRatio  = -1.0f;
	float thisRatio = -1.0f;
	float re_out;

	int thisIO = 0;
	int thisrecall = 0;
	int allIO  = 0;
	int allrecall = 0;
	
	clock_t startTime = (clock_t) -1.0;
	clock_t endTime   = (clock_t) -1.0;
									// init the results
	g_memory += (long) sizeof(ResultItem) * maxk;
	ResultItem* rslt = new ResultItem[maxk];
	for (i = 0; i < maxk; i++) {
		rslt[i].id_ = -1;
		rslt[i].dist_ = MAXREAL;
	}

	QALSH* lsh = new QALSH();		// restore QALSH
	if (lsh->restore(output_folder, n_ring)) {        //
		error("Could not restore qalsh\n", true);
	}

	char output_set[200];
	strcpy(output_set, output_folder);
	strcat(output_set, "L2_qalsh.out");

	fp = fopen(output_set, "w");	// open output file
	if (!fp) {
		printf("Could not create the output file.\n");
		return 1;
	}

	FILE* f_cen = fopen("center.txt", "r");
	FILE* f_root = fopen("root.txt", "r");
	int ordinal;

	int** Root_I = new int* [m];
	int** upp = new int* [m];
	int** low = new int* [m];
	int* last_p = new int[m];
	
	for(int i = 0; i < m; i++){
	    Root_I[i] = new int[n_ring];
	    upp[i] = new int[n_ring];
		low[i] = new int[n_ring];	
	}
	for(int i = 0; i < m; i++){
		fscanf(f_root, "%d\n", &ordinal);
		for(int k = 0; k < n_ring; k++){
			fscanf(f_root, "%d ", &(Root_I[i][k]));
		}
		
		fscanf(f_root, "%d\n", &(last_p[i]));
		
		for(int k = 0; k < n_ring; k++){
			fscanf(f_root, "%d ", &(low[i][k]));
			fscanf(f_root, "%d ", &(upp[i][k]));
		}
		
		fscanf(f_root, "\n");
	}
	
	fclose(f_root);
	
	float** r_lo = new float*[m];
	float** r_up = new float*[m];
	float** r_lo_sqr = new float*[m];
	float** r_up_sqr = new float*[m];
	float** center = new float*[m];
	
	for(int i  = 0; i < m; i++){
		r_lo[i] = new float[n_ring];
		r_up[i] = new float[n_ring];
		r_lo_sqr[i] = new float[n_ring];
		r_up_sqr[i] = new float[n_ring];
		center[i] = new float[sub_d];	
	}
	
    int count = 0;
	
	for(int i = 0; i < m; i++){
		for(int s = 0; s < sub_d; s++){
			fscanf(f_cen, "%f ", &center[i][s]);
	    }    
	    fscanf(f_cen, "\n");
		for(int s = 0; s < n_ring; s++){
		    fscanf(f_cen, "%f ", &r_lo[i][s]);
			fscanf(f_cen, "%f ", &r_up[i][s]);
			r_lo_sqr[i][s] = r_lo[i][s] * r_lo[i][s];
			r_up_sqr[i][s] = r_up[i][s] * r_up[i][s];
		}
		fscanf(f_cen, "\n");
    }

	fclose(f_cen); f_cen = NULL;

	//----------------------------------------------------------------

	printf("QALSH for c-k-ANN Search: \n");
	printf("    Top-k\tRatio\t\tI/O\t\tTime (ms)\n");
	
	    ratio = c;
	    
	        top_k = K;
		allRatio = 0.0f;
		allIO = 0;
		allrecall = 0;
		
		startTime = clock();
		for (i = 0; i < qn; i++) {
			thisIO = lsh->knn(query[i], top_k, ratio, rslt, output_folder, center, r_lo, r_up, r_lo_sqr, r_up_sqr, Root_I, last_p, low, upp, lam);      //
			thisRatio = 0.0f;
			thisrecall = 0;
			for (j = 0; j < top_k; j++) {
				thisRatio += rslt[j].dist_ / R[i * maxk + j];
			}
			for (j = 0; j < top_k; j++) {
			   for(int j2 = 0; j2 < top_k; j2++){
			     if(rslt[j].id_ == recall[i * maxk + j2])
			       {thisrecall++; break;}
			   } 
			}
				
			thisRatio /= top_k;
			allRatio += thisRatio;
			allIO += thisIO;
			allrecall += thisrecall;
		}
		endTime = clock();
		allTime = ((float) endTime - startTime) / CLOCKS_PER_SEC;

		allRatio = allRatio / qn;
		allTime = (allTime * 1000.0f) / qn;
		allIO = (int) ceil((float) allIO / (float) qn);
		re_out  = (float)allrecall / (float)qn;

		printf("    %3d\t\t%.4f\t\t%d\t\t%.2f\n", top_k, allRatio, 
			allIO, allTime);
		fprintf(fp, "%d\t%f\t%d\t%f\n", top_k, allRatio, allIO, allTime);
		fprintf(f1, "%f,%d,%f,%f,%d,%f\n", ratio, top_k, allRatio, re_out, allIO, allTime);
//	}
//}
	fclose(fp);						// close output file
	fclose(f1);

	// -------------------------------------------------------------------------
	//  Release space
	// -------------------------------------------------------------------------
	if (query != NULL) {			// release <query>
		for (i = 0; i < qn; i++) {
			delete[] query[i]; query[i] = NULL;
		}
		delete[] query; query = NULL;
		g_memory -= SIZEFLOAT * qn * d;
	}
	if (lsh != NULL) {				// release <lsh>
		delete lsh; lsh = NULL;
	}
									// release <R> and (/or) <rslt>
	if (R != NULL || rslt != NULL) {
		delete[] R; R = NULL;
		delete[] rslt; rslt = NULL;
		g_memory -= (SIZEFLOAT * qn * maxk + sizeof(ResultItem) * maxk);
	}

	
	for(int i  = 0; i < m; i++){
		delete[] r_lo[i];
		delete[] r_up[i];
		delete[] r_lo_sqr[i];
		delete[] r_up_sqr[i];
		delete[] center[i];		
	}
	
	delete[] r_lo; r_lo = NULL;
	delete[] r_up; r_up = NULL;
	delete[] r_lo_sqr; r_lo_sqr = NULL;
	delete[] r_up_sqr; r_up_sqr = NULL;
	delete[] center; center = NULL;
	
	in.close();

	return ret;
}

// -----------------------------------------------------------------------------
int linear_scan(					// brute-force linear scan (data in disk)
	int   n,							// number of data points
	int   qn,							// number of query points
	int   d,							// dimension of space
	int   B,							// page size
	char* query_set,					// address of query set
	char* truth_set,					// address of ground truth file
	char* output_folder)				// output folder
{
	// -------------------------------------------------------------------------
	//  Allocation and initialzation.
	// -------------------------------------------------------------------------
	clock_t startTime = (clock_t) -1.0f;
	clock_t endTime   = (clock_t) -1.0f;

	int kNNs[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	int maxRound = 11;
	
	int i, j, top_k;
	int maxk = MAXK;

	float allTime   = -1.0f;
	float thisRatio = -1.0f;
	float allRatio  = -1.0f;

	g_memory += (SIZEFLOAT * (d + d + (qn + 1) * maxk) + SIZECHAR * (600 + B));
	
	float* knn_dist = new float[maxk];
	for (i = 0; i < maxk; i++) {
		knn_dist[i] = MAXREAL;
	}

	float** R = new float*[qn];
	for (i = 0; i < qn; i++) {
		R[i] = new float[maxk];
		for (j = 0; j < maxk; j++) {
			R[i][j] = 0.0f;
		}
	}

	float* data     = new float[d];	// one data object
	float* query    = new float[d];	// one query object

	char* buffer    = new char[B];	// every time can read one page
	char* fname     = new char[200];// file name for data
	char* data_path = new char[200];// data path
	char* out_set	= new char[200];// output file

	// -------------------------------------------------------------------------
	//  Open the output file, and read the ground true results
	// -------------------------------------------------------------------------
	strcpy(out_set, output_folder);	// generate output file
	strcat(out_set, "L2_linear.out");

	FILE* ofp = fopen(out_set, "w");
	if (!ofp) {
		printf("I could not create %s.\n", out_set);
		return 1;
	}
									// open ground true file
	FILE* tfp = fopen(truth_set, "r");
	if (!tfp) {
		printf("I could not create %s.\n", truth_set);
		return 1;
	}
									// read top-k nearest distance
	fscanf(tfp, "%d %d\n", &qn, &maxk);
	for (int i = 0; i < qn; i++) {
		fscanf(tfp, "%d", &j);
		for (j = 0; j < maxk; j ++) {
			fscanf(tfp, " %f", &(R[i][j]));
		}
	}
	fclose(tfp);					// close ground true file

	// -------------------------------------------------------------------------
	//  Calc the number of data object in one page and the number of data file.
	//  <num> is the number of data in one data file
	//  <total_file> is the total number of data file
	// -------------------------------------------------------------------------
	int num = (int) floor((float) B / (d * SIZEFLOAT));
	int total_file = (int) ceil((float) n / num);
	if (total_file == 0) return 1;

	// -------------------------------------------------------------------------
	//  Brute-force linear scan method (data in disk)
	//  For each query, we limit that we can ONLY read one page of data.
	// -------------------------------------------------------------------------
	int count = 0;
	float dist = -1.0F;
									// generate the data path
	strcpy(data_path, output_folder);
	strcat(data_path, "data/");
	 qn = 1;
	printf("Linear Scan Search:\n");
	printf("    Top-k\tRatio\t\tI/O\t\tTime (ms)\n");
	for (int round = 0; round < maxRound; round++) {
		top_k = kNNs[round];
		allRatio = 0.0f;

		startTime = clock();
		FILE* qfp = fopen(query_set, "r");
		if (!qfp) error("Could not open the query set.\n", true);

		for (i = 0; i < qn; i++) {
			// -----------------------------------------------------------------
			//  Step 1: read a query from disk and init the k-nn results
			// -----------------------------------------------------------------
			fscanf(qfp, "%d", &j);
			for (j = 0; j < d; j++) {
				fscanf(qfp, " %f", &query[j]);
			}

			for (j = 0; j < top_k; j++) {
				knn_dist[j] = MAXREAL;
			}

			// -----------------------------------------------------------------
			//  Step 2: find k-nn results for the query
			// -----------------------------------------------------------------
			for (j = 0; j < total_file; j++) {
				// -------------------------------------------------------------
				//  Step 2.1: get the file name of current data page
				// -------------------------------------------------------------
				get_data_filename(j, data_path, fname);

				// -------------------------------------------------------------
				//  Step 2.2: read one page of data into buffer
				// -------------------------------------------------------------
				if (read_buffer_from_page(B, fname, buffer) == 1) {
					error("error to read a data page", true);
				}

				// -------------------------------------------------------------
				//  Step 2.3: find the k-nn results in this page. NOTE: the 
				// 	number of data in the last page may be less than <num>
				// -------------------------------------------------------------
				if (j < total_file - 1) count = num;
				else count = n % num;

				for (int z = 0; z < count; z++) {
					read_data_from_buffer(z, d, data, buffer);
					dist = calc_l2_dist(data, query, d);

					int ii, jj;
					for (jj = 0; jj < top_k; jj++) {
						if (compfloats(dist, knn_dist[jj]) == -1) {
							break;
						}
					}
					if (jj < top_k) {
						for (ii = top_k - 1; ii >= jj + 1; ii--) {
							knn_dist[ii] = knn_dist[ii - 1];
						}
						knn_dist[jj] = dist;
					}
				}
			}

			thisRatio = 0.0f;
			for (j = 0; j < top_k; j++) {
				thisRatio += knn_dist[j] / R[i][j];
			}
			thisRatio /= top_k;
			allRatio += thisRatio;
		}
		// -----------------------------------------------------------------
		//  Step 3: output result of top-k nn points
		// -----------------------------------------------------------------
		fclose(qfp);				// close query file
		endTime  = clock();
		allTime  = ((float) endTime - startTime) / CLOCKS_PER_SEC;
		allTime = (allTime * 1000.0f) / qn;
		allRatio = allRatio / qn;
									// output results
		printf("    %3d\t\t%.4f\t\t%d\t\t%.2f\n", top_k, allRatio, 
			total_file, allTime);
		fprintf(ofp, "%d\t%f\t%d\t%f\n", top_k, allRatio, total_file, allTime);
	}
	printf("\n");
	fclose(ofp);					// close output file

	// -------------------------------------------------------------------------
	//  Release space
	// -------------------------------------------------------------------------
	if (R != NULL) {
		for (i = 0; i < qn; i++) {
			delete[] R[i]; R[i] = NULL;
		}
		delete[] R; R = NULL;
	}
	if (knn_dist != NULL || buffer != NULL || data != NULL || query != NULL) {
		delete[] knn_dist; knn_dist = NULL;
		delete[] buffer; buffer = NULL;
		delete[] data; data = NULL;
		delete[] query; query = NULL;
	}
	if (fname != NULL || data_path != NULL || out_set != NULL) {
		delete[] fname; fname = NULL;
		delete[] data_path; data_path = NULL;
		delete[] out_set; out_set = NULL;
	}
	g_memory -= (SIZEFLOAT * (d + d + (qn + 1) * maxk) + SIZECHAR * (600 + B));
	
	return 0;
}
