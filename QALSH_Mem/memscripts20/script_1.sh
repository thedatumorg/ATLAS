#!/bin/bash

# ------------------------------------------------------------------------------
#  Parameters ('dtype' has 4 options: uint8, uint16, int32, float32)
# ------------------------------------------------------------------------------
cd ../methods/


dname=ukbench
n=1097907
qn=200
d=128
pf=/dataset-path/${dname}
df=/dataset-path/${dname}/

B=4096
c=2.0
dtype=float32
p=2.0
z=0.0
log=log1


of=../results/${dname}/c=${c}_p=${p}/

./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

/usr/bin/time -v ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
-dt ${dtype} -pf ${pf} -of ${of} -dname ${log} &> ../logs/${dname}-${log}.txt


B=4096
c=1.75
dtype=float32
p=2.0
z=0.0
log=log2


of=../results/${dname}/c=${c}_p=${p}/

./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

/usr/bin/time -v ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
-dt ${dtype} -pf ${pf} -of ${of} -dname ${log} &> ../logs/${dname}-${log}.txt



B=4096
c=1.5
dtype=float32
p=2.0
z=0.0
log=log3


of=../results/${dname}/c=${c}_p=${p}/

./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

/usr/bin/time -v ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
-dt ${dtype} -pf ${pf} -of ${of} -dname ${log} &> ../logs/${dname}-${log}.txt



B=4096
c=1.25
dtype=float32
p=2.0
z=0.0
log=log4


of=../results/${dname}/c=${c}_p=${p}/

./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

/usr/bin/time -v ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
-dt ${dtype} -pf ${pf} -of ${of} -dname ${log} &> ../logs/${dname}-${log}.txt


B=2048
c=2.0
dtype=float32
p=2.0
z=0.0
log=log5


of=../results/${dname}/c=${c}_p=${p}/

./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

/usr/bin/time -v ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
-dt ${dtype} -pf ${pf} -of ${of} -dname ${log} &> ../logs/${dname}-${log}.txt
