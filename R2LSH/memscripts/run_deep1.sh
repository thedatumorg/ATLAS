# rm ./src/*.o

cd /path/build


#----------parameters-----------#
n=1000000
d=256
m=40
ring=30
qn=200
lam=0.7
_B=4096
C=1.3

K=20


folderPath=deep
dPath=base.fvecs
qPath=query.fvecs
rPath=truth.bin
idPath=recall.txt
index=deep_1



dPath=/dataset-path/${folderPath}/${dPath}
qPath=/dataset-path/${folderPath}/${qPath}
rPath=/dataset-path/${folderPath}/${rPath}
idPath=/dataset-path/${folderPath}/${idPath}
indexes=/index-path/${index}

rm -rf ${indexes}
mkdir ${indexes}

#------------generate ground truth---------------------
./r2lsh -alg 0 -n ${n} -qn ${qn} -d ${d} -ds ${dPath} -qs ${qPath} -ts ${rPath} -rs ${idPath} 


#------------build index-----------------------------------
/usr/bin/time -v ./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B ${_B} -ds ${dPath} -of ${indexes} &> ../logs/${folderPath}-index-1-search-0.txt


#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-1-search-1.txt


C=1.15

#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-1-search-2.txt

C=1.6

#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-1-search-3.txt

C=1.3

lam=0.6
#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-1-search-4.txt

lam=0.8
#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-1-search-5.txt

lam=0.7

m=20

rm -rf ${indexes}
mkdir ${indexes}


#------------build index-----------------------------------
/usr/bin/time -v ./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B ${_B} -ds ${dPath} -of ${indexes} &> ../logs/${folderPath}-index-2-search-0.txt


#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-2-search-1.txt

m=60

rm -rf ${indexes}
mkdir ${indexes}


#------------build index-----------------------------------
/usr/bin/time -v ./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B ${_B} -ds ${dPath} -of ${indexes} &> ../logs/${folderPath}-index-3-search-0.txt


#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-3-search-1.txt

m=40

ring=15

rm -rf ${indexes}
mkdir ${indexes}


#------------build index-----------------------------------
/usr/bin/time -v ./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B ${_B} -ds ${dPath} -of ${indexes} &> ../logs/${folderPath}-index-4-search-0.txt


#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-4-search-1.txt

ring=60

rm -rf ${indexes}
mkdir ${indexes}


#------------build index-----------------------------------
/usr/bin/time -v ./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B ${_B} -ds ${dPath} -of ${indexes} &> ../logs/${folderPath}-index-5-search-0.txt


#------------KNN search----------------------------------------
/usr/bin/time -v ./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ${rPath} -rs ${idPath} -of ${indexes} -lam ${lam} -topK ${K} -c ${C} &> ../logs/${folderPath}-index-5-search-1.txt

ring=30

