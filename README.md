# DCSR
The artifacts for our paper "DCSR: A Fast Data Structure with Leaf-Oriented Locks for Streaming Graph Processing" (EDBT 2026). The primary innovation of DCSR is a leaf-oriented parallel update strategy that comprises two phases of lock-based update and decoupled rebalancing to fully exploit parallelism with minimal conflicts. We developed DCSR based on [BYO](https://github.com/wheatman/BYO), a unified framework for benchmarking large-scale graph containers.  

## Description
The files or folders involving DCSR in our supplemental material include

- DCSR/benchmarks/run_structures/run_dcsr.cc. This file provides the interfaces for streaming graph processing to BYO. 

- DCSR/benchmarks/run_structures/BUILD. We add the dependencies for DCSR to the original BUILD file. 

- DCSR/WORKSPACE. We add the DCSR local repository to the original WORKSPACE file. 

- DCSR/external/dcsr. This folder contains the main file \texttt{DCSR/external/dcsr/include/dcsr1.0/partition.h} and other files for DCSR.

\end{itemize}
