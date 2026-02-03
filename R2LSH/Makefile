SRCS= src/gendef.cpp src/blk_file.cpp src/util.cpp src/random.cpp src/block_file.cpp src/b_node.cpp \
src/b_tree.cpp src/qalsh.cpp src/ann.cpp src/main.cpp

OBJS=$(SRCS:.cpp=.o)

CXX?=g++ -std=c++11
CPPFLAGS=-w -O3 -I /usr/local/boost_1_75_0/

.PHONY: clean

all: $(OBJS)
	$(CXX) -o r2lsh $(OBJS)


gendef.o: src/gendef.h

blk_file.o: src/blk_file.h

util.o: src/util.h

random.o: src/random.h

block_file.o: src/block_file.h

b_node.o: src/b_node.h

b_tree.o: src/b_tree.h

qalsh.o: src/qalsh.h

ann.o: src/ann.h

main.o:

clean:
	-rm $(OBJS) r2lsh
