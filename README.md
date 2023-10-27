Communication-Avoiding 3D Dense Matrix Matrix Multiplication
============================================================
[High Performance Computing task, University of Warsaw 2022/23](https://www.mimuw.edu.pl/~krzadca/mpi-labs/ca3dmm.html)

1 Introduction
--------------

As communication can easily dominate computation in HPC, there is a significant interest in communication-avoiding or communication-optimal algorithms: algorithms that are proved to complete with minimal communication.
The goal of this project is to implement a very recent addition to this class of algorithms, CA3DMM. CA3DMM solves a classic problem of dense matrix multiplication.

2 CA3DMM description
--------------------

The CA3DMM algorithm takes a 3D perspective on the problem. Denote $C=AB$, where $A$ is a $m$-row by $k$-column matrix; and $B$ is a $k$-row by $n$-column matrix.

The processors are partitioned into a 3D configuration, $p_m \times p_n \times p_k$. Matrix $C$ is partitioned into $p_m \times p_n$ grid. Processes in the $k$ dimension shard between each other the multiplications to compute a single element of $C$; and then aggregate the result.

There two main ideas in the algorithm: (1) pick $p_m$, $p_n$ and $p_k$ to minimize the communication volume ($\min p_m kn + p_n mk + p_k mn$, equation 4 in the paper); (2) use the Cannon algorithm to perform the matrix-matrix multiplications. The second idea is somewhat tricky, as the Cannon algorithm requires a square processor grid, but the communication-optimal solution does not necessarily have a square grid, $p_m = p_n$.

The CA3DMM algorithm works as follows:

1.  Solve $\min p_m kn + p_n mk + p_k mn$ with constraints: (a) $l p \leq p_m p_n p_k \leq p$ ($l$ is a constant, e.g. 0.95; this ensures we use as many processors as possible, but not necessarily all processors available); and (b) $\mod (\max(p_m, p_n), \min(p_m, p_n))=0$ (this ensures the Cannon algorithm can be performed). Among admissible, optimal solutions, pick the one using as many processors as possible, thus maximizing $p_m p_n p_k$.
2.  Organize processes into $p_k$ groups, each group has $p_m \times p_n$ processes.
3.  Assign $A$ to $p_k$ groups in a block-column order; assign $B$ to $p_k$ groups in a block-row order. Denote by $A_i$ the block-column of A assigned to group $i$; and by $B_i$ the block-row of B assigned to group $i$.
4.  Within the $i$-th group: organize processes into $c = \max(p_m, p_n) / \min(p_m, p_n)$ Cannon groups. For example, assume $p_n > p_m$. $B_i$ is then not replicated. $A_i$ is replicated c times — each Cannon group stores its complete copy of $A_i$.
5.  Perform the Cannon's algorithm in each Cannon group and in each of the $p_k$ groups to get $C_i$.
6.  Reduce $C = \sum_{i=1}^k C_i$.

3 Specific requirements
-----------------------

*   $A$ and $B$ are generated pseudo-randomly; an interface and an example implementation of the generator is provided, but you may switch the implementation.
*   Each element of $A$ and $B$ is generated only once. Thus, in order to replicate $A$ or $B$ (when $c > 1$), processes need to communicate between each other (rather than place redundant calls to the generator).
*   $C$, the result, is not redistributed so that it is stored not redundantly.

4 Execution
------------------

The program can be run using the following instructions:

`rm -rf build; mkdir build; cd build; cmake ..; make`

`srun ./ca3dmm n m k -s seeds [-g ge_value] [-v]`

where:

*   `n m k` are the dimensions of the matrices (positive integers, each smaller than $10^6$).
*   `-s seeds` is a comma-separated list of N pairs of positive integer seeds (e.g.: `-s 7,42,13,15`); the program will perform `N` multiplications in total; use the first element of i-th pair to initialize $A$, and the second to initialize $B$ (e.g. in the second multiplication, it will initialize $A$ with 13 and $B$ with 15). We do N multiplications to amortize the costs of starting the MPI program.
*   `-v` prints the resulting matrices $C$ (the multiplication results) in the row-major order. For each matrix, the first line specifies the number of rows and the number of columns of the result (space-separated); i+1-th line is the i-th row of the matrix (space-separated). The whole $C$ may not fit in the memory of a single process — but this printing does not need to be perfromance-optimized.
*   `-g ge_value` for each resulting matrix $C$, prints the number of elements in $C$ greater than or equal to the `ge_value`. `-g` and `-v` are mutually-exclusive.

The assumption is that the order of arguments will be exactly as above; and that the input is correct (no need to check the input parameters).


5 Solution content
------------------

*   `densematgen.h`: random matrix generator interface. This file cannot be modified.
*   `densematgen.cpp`: random matrix generator implementation. You might use a different implementation of the generator (but it should be stateless);
*   `CMakeLists.txt` as we will use CMake for building the solution.
*   `report.pdf`: a report describing the implementation - estimating the numerical intensity of the problem (as in the roofline model), describing the implemented optimizations, showing weak and strong scaling results.

6 Resources
----------------------

*   [https://huanghua1994.github.io/files/SC22-Huang-Chow.pdf](https://huanghua1994.github.io/files/SC22-Huang-Chow.pdf)

