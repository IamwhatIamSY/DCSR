# DCSR
The artifacts for our paper "DCSR: A Fast Data Structure with Leaf-Oriented Locks for Streaming Graph Processing" (EDBT 2026). The primary innovation of DCSR is a leaf-oriented parallel update strategy that comprises two phases of lock-based update and decoupled rebalancing to fully exploit parallelism with minimal conflicts. We developed DCSR based on [BYO](https://github.com/wheatman/BYO), a unified framework for benchmarking large-scale graph containers.  

## Description
The files or folders for DCSR in our artifacts include:

- run_dcsr.cc. This file provides the interfaces for streaming graph processing to BYO. 
- BUILD. We add the dependencies for DCSR to the original BUILD file. 
- WORKSPACE. We add the DCSR local repository to the original WORKSPACE file. 
- dcsr. This folder contains the main file dcsr/include/dcsr1.0/partition.h and other files for DCSR.

## Dependencies
The dependencies for [BYO](https://github.com/wheatman/BYO) should be satisfied:

- Bazel 2.1.0. 
- g++ >= 11 with OpenMP support.
- Other external libraries as submodules acquired by [BYO](https://github.com/wheatman/BYO).

## Installation
 First of all, download the source code of [BYO](https://github.com/wheatman/BYO) and DCSR.
 
```
git clone https://github.com/wheatman/BYO.git
git clone https://github.com/IamwhatIamSY/DCSR.git
```
 
 Then, the files of DCSR should be integrated into BYO following the steps:

- add the file DCSR/run_dcsr.cc to the folder BYO/benchmarks/run_structures/.
  
```
mv DCSR/run_dcsr.cc BYO/benchmarks/run_structures/
```

- replace the file BYO/benchmarks/run_structures/BUILD with the file DCSR/BUILD.

```
mv BYO/benchmarks/run_structures/BUILD BYO/benchmarks/run_structures/BUILD.bak
mv DCSR/BUILD BYO/benchmarks/run_structures/
```

- replace the file BYO/WORKSPACE with the file DCSR/WORKSPACE.

```
mv BYO/WORKSPACE BYO/WORKSPACE.bak
mv DCSR/WORKSPACE BYO/
```

- add the folder DCSR/dcsr to the folder BYO/external/.

```
mv DCSR/dcsr BYO/external/
```

## Compilation
Compile the files as required by [BYO](https://github.com/wheatman/BYO).

## How to run
DCSR can be run as required by [BYO](https://github.com/wheatman/BYO). Here are some examples:

- DCSR can be built with OpenMP for parallelism.
```
bazel build benchmarks/run_structures:run_dcsr --config=openmp
```

- Use DCSR for graph computation on an undirected graph with source vertex 10.
```
numactl -i all ./bazel-bin/benchmarks/run_structures/run_dcsr -s -src 10 /path/to/graph
```

- Use DCSR for graph update on an undirected graph.
```
numactl -i all ./bazel-bin/benchmarks/run_structures/run_dcsr -i -s /path/to/graph
```
