# taskset --cpu-list 68-68 bash peram-search-sun.sh
declare -a arr=("VAQ256m32min6max12var1" "VAQ256m32min5max12var1" "VAQ256m32min6max11var1" "VAQ256m32min5max13var1")


NNSTRATEGY=EA_TI1000

KMEANS_VER=0 
DATASET=yahoo-minilm-384-normalized
DATASET_PATH=../../../dataset
BIN_PATH=../build/examples

SIFT_PATH=$DATASET_PATH/$DATASET
DATASET_SIZE=677305
QUERIES_SIZE=100
DIM=384
knn=100
REFINE=100,200,300,400,500,600,700,800,900,1000
TEST_ID=$DATASET-bulk-testing-27


SP_DIM=1
SP_SUB_SIZE=1
CUM_VAR=0.95
MIN_DIMS=64

MIN_DIMS_PER_SUBSPACE=2




CUM_VAR=0.9

VISIT_CLUSTER=0.1

for VAQPARAM in "${arr[@]}"
do
  METHOD=${VAQPARAM},${NNSTRATEGY}
  /usr/bin/time -v $BIN_PATH/demo_vaq --dataset $SIFT_PATH/base.fvecs\
    --queries $SIFT_PATH/query.fvecs\
    --file-format-ori fvecs\
    --timeseries-size $DIM\
    --dataset-size $DATASET_SIZE\
    --queries-size $QUERIES_SIZE\
    --result ukbench.csv\
    --groundtruth $SIFT_PATH/groundtruth.ivecs\
    --groundtruth-format ivecs\
    --method $METHOD\
    --k $knn\
    --refine $REFINE\
    --test-id ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}\
    --visit-cluster $VISIT_CLUSTER\
    --new-alg 22\
    --exp-var-multiplier 2.0\
    --extra-bit -4\
    --sub-space-swapping 1\
    --kmeans-ver $KMEANS_VER\
    --nonUniformSubspaces 1\
    --fixedSamplingForKmeans 1\
    --spSubSize $SP_SUB_SIZE\
    --spDim $SP_DIM\
    --cumVar $CUM_VAR\
    --minDimsPerSubspace $MIN_DIMS_PER_SUBSPACE\
    --storeResults 1\ &> ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}-mem.txt
    
done


VISIT_CLUSTER=0.075


for VAQPARAM in "${arr[@]}"
do
  METHOD=${VAQPARAM},${NNSTRATEGY}
  /usr/bin/time -v $BIN_PATH/demo_vaq --dataset $SIFT_PATH/base.fvecs\
    --queries $SIFT_PATH/query.fvecs\
    --file-format-ori fvecs\
    --timeseries-size $DIM\
    --dataset-size $DATASET_SIZE\
    --queries-size $QUERIES_SIZE\
    --result ukbench.csv\
    --groundtruth $SIFT_PATH/groundtruth.ivecs\
    --groundtruth-format ivecs\
    --method $METHOD\
    --k $knn\
    --refine $REFINE\
    --test-id ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}\
    --visit-cluster $VISIT_CLUSTER\
    --new-alg 22\
    --exp-var-multiplier 2.0\
    --extra-bit -4\
    --sub-space-swapping 1\
    --kmeans-ver $KMEANS_VER\
    --nonUniformSubspaces 1\
    --fixedSamplingForKmeans 1\
    --spSubSize $SP_SUB_SIZE\
    --spDim $SP_DIM\
    --cumVar $CUM_VAR\
    --minDimsPerSubspace $MIN_DIMS_PER_SUBSPACE\
    --storeResults 1\ &> ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}-mem.txt
    
done



CUM_VAR=0.95

VISIT_CLUSTER=0.1

for VAQPARAM in "${arr[@]}"
do
  METHOD=${VAQPARAM},${NNSTRATEGY}
  /usr/bin/time -v $BIN_PATH/demo_vaq --dataset $SIFT_PATH/base.fvecs\
    --queries $SIFT_PATH/query.fvecs\
    --file-format-ori fvecs\
    --timeseries-size $DIM\
    --dataset-size $DATASET_SIZE\
    --queries-size $QUERIES_SIZE\
    --result ukbench.csv\
    --groundtruth $SIFT_PATH/groundtruth.ivecs\
    --groundtruth-format ivecs\
    --method $METHOD\
    --k $knn\
    --refine $REFINE\
    --test-id ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}\
    --visit-cluster $VISIT_CLUSTER\
    --new-alg 22\
    --exp-var-multiplier 2.0\
    --extra-bit -4\
    --sub-space-swapping 1\
    --kmeans-ver $KMEANS_VER\
    --nonUniformSubspaces 1\
    --fixedSamplingForKmeans 1\
    --spSubSize $SP_SUB_SIZE\
    --spDim $SP_DIM\
    --cumVar $CUM_VAR\
    --minDimsPerSubspace $MIN_DIMS_PER_SUBSPACE\
    --storeResults 1\ &> ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}-mem.txt
    
done


VISIT_CLUSTER=0.075


for VAQPARAM in "${arr[@]}"
do
  METHOD=${VAQPARAM},${NNSTRATEGY}
  /usr/bin/time -v $BIN_PATH/demo_vaq --dataset $SIFT_PATH/base.fvecs\
    --queries $SIFT_PATH/query.fvecs\
    --file-format-ori fvecs\
    --timeseries-size $DIM\
    --dataset-size $DATASET_SIZE\
    --queries-size $QUERIES_SIZE\
    --result ukbench.csv\
    --groundtruth $SIFT_PATH/groundtruth.ivecs\
    --groundtruth-format ivecs\
    --method $METHOD\
    --k $knn\
    --refine $REFINE\
    --test-id ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}\
    --visit-cluster $VISIT_CLUSTER\
    --new-alg 22\
    --exp-var-multiplier 2.0\
    --extra-bit -4\
    --sub-space-swapping 1\
    --kmeans-ver $KMEANS_VER\
    --nonUniformSubspaces 1\
    --fixedSamplingForKmeans 1\
    --spSubSize $SP_SUB_SIZE\
    --spDim $SP_DIM\
    --cumVar $CUM_VAR\
    --minDimsPerSubspace $MIN_DIMS_PER_SUBSPACE\
    --storeResults 1\ &> ${TEST_ID},${VAQPARAM},${NNSTRATEGY},${SP_DIM}-${MIN_DIMS_PER_SUBSPACE}-${MIN_REDUCTION_THRESHOLD}-${CUM_VAR}-${VISIT_CLUSTER}-mem.txt
    
done











