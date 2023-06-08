// #include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <densematgen.h>
#include <vector>

#include <bits/stdc++.h>

#define ll long long

const double l = 0.95;

void opt_grid_dim(int P, int n, int m, int k, int& Pn, int& Pm, int& Pk) {
    long long opt = -1;
    for (int pn = 1; pn * pn <= P; pn++) {
        for (int pm = pn; pm * pn <= P; pm += pn) {
            for (int pk = ceil((l * P) / (pm * pn)); pk * pm * pn <= P; pk++) {
                long long target;
                // Pm is multiplication of Pn
                target = (ll)(pm) * (ll)(k) * (ll)(n) + (ll)(pn) * (ll)(m) * (ll)(k) + (ll)(pk) * (ll)(m) * (ll)(n);
                if (opt == -1 || target <= opt) {
                    opt = target;
                    Pn = pn;
                    Pm = pm;
                    Pk = pk;
                }
                // Pn is multiplication of Pm
                target = (ll)(pn) * (ll)(k) * (ll)(n) + (ll)(pm) * (ll)(m) * (ll)(k) + (ll)(pk) * (ll)(m) * (ll)(n);
                if (opt == -1 || target <= opt) {
                    opt = target;
                    Pn = pm;
                    Pm = pn;
                    Pk = pk;
                }
            }
        }
    }
    if (opt == -1) {
        Pn = 0;
        Pm = 0;
        Pk = 0;
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc,&argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 4) {
        if (rank == 0)
            std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\n";
        MPI_Finalize();
        return 1;
    }

    // long long n = atoll(argv[1]);
    // long long m = atoll(argv[2]);
    // long long k = atoll(argv[3]);
    
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);

    std::string seeds = "";
    int ge_value = -1;
    bool vprint = false;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0) {
            if (i < argc - 1) {
                i++;
                seeds = argv[i];
            }
            else {
                if (rank == 0)
                    std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\n";
                MPI_Finalize();
                return 1;                
            }
        }
        else if (strcmp(argv[i], "-g") == 0) {
            if (i < argc - 1) {
                i++;
                ge_value = atoi(argv[i]);
            }
            else {
                if (rank == 0)
                    std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\n";
                MPI_Finalize();
                return 1;                
            }
        }
        else if (strcmp(argv[i], "-v") == 0) {
            vprint = true;
        }
        else {
            if (rank == 0)
                std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\n";
            MPI_Finalize();
            return 1;                
        }
    }

    if (seeds == "") {
        if (rank == 0)
            std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\n";
        MPI_Finalize();
        return 1;      
    }

    std::vector<int> seedsA, seedsB;
    std::stringstream ss(seeds);
    int sA, sB;
    std::string substrA, substrB;

    while (ss.good()) {
        std::getline(ss, substrA, ',');
        if (ss.good()) {
            std::getline(ss, substrB, ',');
        }
        else {
            if (rank == 0)
                std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\nError: even number of seeds\n";
            MPI_Finalize();
            return 1;
        }

        try {
            sA = stoi(substrA);
            sB = stoi(substrB);
        } catch(std::exception) {
            if (rank == 0)
                std::cerr << "Usage: ./ca3dmm n m k -s seeds [-g ge_value] [-v]\nError: seed is not a number\n";
            MPI_Finalize();
            return 1;
        }

        seedsA.push_back(sA);
        seedsB.push_back(sB);
    }
    
    int Pn, Pm, Pk;
    opt_grid_dim(P, n, m, k, Pn, Pm, Pk);

    if (rank == 0)
        std::cout << Pn << " " << Pm << " " << Pk << "\n";

    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, (rank < Pn * Pm * Pk ? 1 : 0), rank, &active);

    if (rank >= Pn * Pm * Pk) {
        MPI_Finalize();
        return 0;
    }

    for (int mul = 0; mul < seedsA.size(); mul++) {
        sA = seedsA[mul];
        sB = seedsB[mul];
    
        int slice = rank / (Pm * Pn);
        int row = (rank % (Pm * Pn)) / Pn;
        int col = (rank % (Pm * Pn)) % Pn;

        int dim_k = slice < k % Pk ? (k + Pk - 1) / Pk : k / Pk;
        int dim_m = row < m % Pm ? (m + Pm - 1) / Pm : m / Pm;
        int dim_n = col < n % Pn ? (n + Pn - 1) / Pn : n / Pn;

        int offset_k = slice * (k / Pk) + std::min(slice, k % Pk);
        int offset_m = row * (m / Pm) + std::min(row, m % Pm);
        int offset_n = col * (n / Pn) + std::min(col, n % Pn);

        MPI_Comm row_slice_comm; // ten sam row w tym samym slice
        MPI_Comm_split(active, row * Pk + slice, col, &row_slice_comm);

        // int x;
        // MPI_Comm_size(row_slice_comm, &x);
        // if (col == 0) std::cout << x << "\n";

        // if (col == 0) {
        //     std::cout << "(" << row << ", " << slice << ")\n";
        // }

        std::vector<double> A(dim_k * dim_m, 0);
        
        if (col == 0) { // Generuję kawałek macierzy A
            for (int i = 0; i < dim_m; i++) {
                for (int j = 0; j < dim_k; j++) {  
                    A[i * dim_k + j] = generate_double(sA, i + offset_m, j + offset_k);
                    //std::cout << A[i * dim_k + j] << ", ";
                }
                //std::cout << "\n";
            }
        }

        MPI_Bcast(
            &A[0],  /* the message will be written here */
                    /* if my_rank==root, the message will be read from here */
            dim_m * dim_k,  /* number of items in the message */
            MPI_DOUBLE, /* type of data in the message */
            0,   /* if my_rank==root, I'm sending, otherwise I'm receiving */
            row_slice_comm  /* communicator to use */
        );

        // if (col == Pn - 1) {
        //     for (int i = 0; i < dim_m; i++) {
        //         for (int j = 0; j < dim_k; j++) {  
        //             std::cout << A[i * dim_k + j] << ", ";
        //         }
        //         std::cout << "\n";
        //     }
        // }

        MPI_Comm col_slice_comm; // ten sam col w tym samym slice
        MPI_Comm_split(active, col * Pk + slice, row, &col_slice_comm);

        std::vector<double> B(dim_k * dim_n, 0);

        if (row == 0) { // Generuję kawałek macierzy B
            for (int i = 0; i < dim_n; i++) {
                for (int j = 0; j < dim_k; j++) {
                    B[i * dim_k + j] = generate_double(sB, j + offset_k, i + offset_n);
                    // std::cout << B[i * dim_k + j] << " ";
                }
            }
        }

        MPI_Bcast(
            &B[0],  /* the message will be written here */
                    /* if my_rank==root, the message will be read from here */
            dim_n * dim_k,  /* number of items in the message */
            MPI_DOUBLE, /* type of data in the message */
            0,   /* if my_rank==root, I'm sending, otherwise I'm receiving */
            col_slice_comm  /* communicator to use */
        );

        // if (row == Pm - 1) {
        //     for (int i = 0; i < dim_n; i++) {
        //         for (int j = 0; j < dim_k; j++) {  
        //             std::cout << B[i * dim_k + j] << ", ";
        //         }
        //     }
        // }        

        std::vector<double> C(dim_m * dim_n, 0);

        for (int i = 0; i < dim_m; i++) {
            for (int j = 0; j < dim_n; j++) {
                for (int x = 0; x < dim_k; x++) {
                    C[i * dim_n + j] += A[i * dim_k + x] * B[j * dim_k + x];
                }
            }
        }

        MPI_Comm row_col_comm; // ten sam row i col
        MPI_Comm_split(active, row * Pn + col, slice, &row_col_comm);

        std::vector<double> R(dim_m * dim_n, 0);

        // for (int i = 0; i < dim_m; i++) {
        //     for (int j = 0; j < dim_n; j++) {        
        //         MPI_Reduce(&C[i * dim_n + j], &R[i * dim_n + j], 1, MPI_DOUBLE, MPI_SUM, 0,
        //         row_col_comm);
        //     }
        // }

        MPI_Reduce(&C[0], &R[0], dim_n * dim_m, MPI_DOUBLE, MPI_SUM, 0,
                row_col_comm);

        // if (rank == 0) {
        //     std::cout << slice << " " << row << " " << col << "?\n";
        //     std::cout << dim_k << " " << dim_m << " " << dim_n << "?\n";
        // }

        if (slice == 0 && row == 0 && col == 0) {
            for (int i = 0; i < dim_m; i++) {
                for (int j = 0; j < dim_n; j++) {
                    std::cout << R[i * dim_n + j] << " ";
                }
                std::cout << "\n";
            }
        }


    } 


    MPI_Finalize();
    return 0;
    
}