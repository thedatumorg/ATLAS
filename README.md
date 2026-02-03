<h1 align="center">ATLAS</h1>
<h2 align="center">ATLAS: The Landscape of Approximate Similarity Search — Two Decades of Algorithmic Advances</h2>

<h2 id="overview"> 📄 Overview </h2>
Similarity search methods enable efficient retrieval of vectors similar to a given query and play a central role in a wide range of applications. Among the variants, approximate similarity search methods offer high accuracy with substantially improved efficiency over exact methods. Despite substantial progress, existing studies suffer from major limitations: (i) omission of key algorithmic families; (ii) overlooking recent methodological advances; (iii) lack of rigorous statistical validation; and (iv) evaluation on limited datasets reflecting modern AI applications. To address these gaps, we introduce ATLAS, the most comprehensive benchmark of approximate nearest neighbor search methods to date. Specifically, our contributions are fourfold: (i) a systematic review of five major algorithmic categories; (ii) a large-scale evaluation of 45 methods across 58 datasets; (iii) the introduction of a new measure that captures latency over a recall range, offering a threshold-free, unbiased assessment of query efficiency; and (iv) statistical analysis to ensure the robustness of the conclusions.Our findings reveal seven key insights: (i) modern quantization-based methods achieve query efficiency comparable to graph-based algorithms while requiring substantially less memory; (ii) across four categories, previously unreported top performers emerge, with two showing statistically significant improvements; (iii) relative algorithm rankings exhibit variation across data modalities and vector dimensionality; (iv) parameter settings do not consistently transfer across datasets, and performance is highly sensitive to data characteristics; (v) hardware-accelerated methods exhibit architecture-dependent performance; (vi) performance is highly sensitive to implementation quality; and (vii) both indexing strategies and hardware acceleration yield substantial throughput gains at the cost of reduced accuracy. Collectively, these findings sharpen our understanding of the ANNS landscape, uncover previously unexplored behaviors, and guide future research.

<h3 id="dataset">🗄️ Dataset</h3>
Due to limitations in the upload size on GitHub, we host the datasets at a different location. Please download the datasets using the following links

* https://drive.google.com/drive/folders/1olm5AcZ1h7hBEuWEk3UwOWkxs6IXoewB?usp=sharing


<h3 id="algorithms">⚙️ Algorithms</h3>

| Method | Folder
|:------|:-------------|
| [SymphonyQG](https://github.com/gouyt13/SymphonyQG) | SymphonyQG |
| [VSAG](https://github.com/antgroup/vsag) | vsag |
| [SVS-LVQ](https://github.com/intel/ScalableVectorSearch) | ScalableVectorSearch-0.0.2 |
| [SVS](https://github.com/intel/ScalableVectorSearch) | ScalableVectorSearch-0.0.2 |
| [GLASS-HNSW](https://github.com/zilliztech/pyglass) | pyglass |
| [GLASS-NSG](https://github.com/zilliztech/pyglass) | pyglass |
| [FLATNAV](https://github.com/BlaiseMuhirwa/flatnav) | flatnav |
| [FINGER](https://github.com/Patrick-H-Chen/FINGER) | FINGER |
| [DISKANN](https://github.com/microsoft/DiskANN) | DiskANN |
| [NSSG](https://github.com/ZJULearning/SSG) | SSG |
| [HCNNG](https://github.com/Lsyhprum/WEAVESS) | WEAVESS |
| [NSG](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [DPG](https://github.com/Lsyhprum/WEAVESS) | WEAVESS |
| [HNSW-PECOS](https://github.com/Patrick-H-Chen/FINGER) | FINGER |
| [HNSW](https://github.com/facebookresearch/faiss) | faiss-1.7.3 | 
| [EFANNA](https://github.com/Lsyhprum/WEAVESS) | WEAVESS |
| [BOLT](https://github.com/dblalock/bolt) | bolt |
| [VAQ](https://github.com/TheDatumOrg/VAQ) | VAQ |
| [PQFS](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [OPQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [ITQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [PQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [DB-LSH](https://github.com/Jacyhust/DB-LSH) | DB-LSH |
| [PM-LSH](https://github.com/Jacyhust/PM-LSH) | PM-LSH |
| R2-LSH | R2LSH |
| [LCCS-LSH](https://github.com/1flei/lccs-lsh) | lccs-lsh |
| [AWS-LSH](https://github.com/1flei/aws_alsh) | aws_alsh |
| [QALSH](https://github.com/HuangQiang/QALSH_Mem) | QALSH_Mem |
| [C2LSH](https://github.com/1flei/lccs-lsh) | lccs-lsh |
| [SPTAG-KDT](https://github.com/microsoft/SPTAG) | SPTAG |
| [SPTAG-BKT](https://github.com/microsoft/SPTAG) | SPTAG |
| [SCANN](https://github.com/google-research/google-research/tree/master/scann) | scann |
| [MRPT](https://github.com/vioshyvo/mrpt) | mrpt |
| [ANNOY](https://github.com/spotify/annoy) | annoy |
| [FLANN](https://github.com/flann-lib/flann) | flann |
| [KD-TREE](https://github.com/flann-lib/flann) | flann |
| [VP_TREE](https://github.com/nmslib/nmslib) | nmslib |
| [DUMPY-FUZZY](https://github.com/DSM-fudan/Dumpy) | Dumpy |
| [DUMPY](https://github.com/DSM-fudan/Dumpy) | Dumpy |
| [ISAX2+](https://github.com/karimaechihabi/lernaean-hydra) | lernaean-hydra |
| [DS_TREE](https://github.com/karimaechihabi/lernaean-hydra) | lernaean-hydra |
| [VA+FILE](https://github.com/karimaechihabi/lernaean-hydra) | lernaean-hydra |
| [IVFPQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [IMI-OPQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [IMI-PQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library) | RaBitQ-Library |

### ✉️ Contact

If you have any questions or suggestions, feel free to contact:
* Ahmed Samin Yeaser Kabir (kabir.36@osu.edu)
* John Paparrizos (paparrizos.1@osu.edu)
