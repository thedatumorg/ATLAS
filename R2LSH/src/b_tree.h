#ifndef __B_TREE_H
#define __B_TREE_H


class BlockFile;
class BNode;

struct HashValue;
struct Elem;
// -----------------------------------------------------------------------------
//  BTree: b-tree to index hash tables produced by qalsh
// -----------------------------------------------------------------------------
class BTree {
public:
	int root_;						// address of disk for root
	
	BlockFile* file_;				// file in disk to store
	BNode* root_ptr_;				// pointer of root
	int ring;            //the number of ring

	BTree();						// constructor
	~BTree();						// destructor

	void init(						// init a new b-tree
		char* fname,					// file name
		int b_length,					// block length
		int n_ring_);

	void init_restore(				// load an exist b-tree
		char* fname,
		int ring);					// file name

	int bulkload(					// bulkload b-tree from hash table in mem
		HashValue* a_points,			// hash table
		Ring* med,							// number of entries
    	int i,
	    int* Root_I,
		int* last_ptr,
		int* upp,
		int* low);
	

private:
	int read_header(				// read <root> from buffer
		char* buf);						// the buffer

	int write_header(				// write <root> into buffer
		char* buf);						// the buffer (return)

	void load_root();				// load root of b-tree

	void delete_root();				// delete root of b-tree
};

#endif
