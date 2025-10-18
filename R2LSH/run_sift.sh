rm ./src/*.o

#----------parameters-----------#
n=100000000
d=128
m=40
ring=30
qn=100
lam=0.7
dPath=./${dPath}
qPath=./${qPath}

#------------generate ground truth---------------------
./r2lsh -alg 0 -n ${n} -qn ${qn} -d ${d} -ds ${dPath} -qs ${qPath} -ts ./truth.txt 


#------------build index-----------------------------------
./r2lsh -alg 1 -n ${n} -d ${d} -sd 2 -m ${m} -ring ${ring} -B 4096 -ds ${dPath} -of ./results/


#------------KNN search----------------------------------------
./r2lsh -alg 2 -qn ${qn} -d ${d} -sd 2 -m ${m} -ring ${ring} -qs ${qPath} -ts ./truth.txt -of ./results/ -lam ${lam} -topK 100 -c 1.3






















