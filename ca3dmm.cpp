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

        int k_ceil = (k + Pk - 1) / Pk;
        int m_ceil = (m + Pm - 1) / Pm;
        int n_ceil = (n + Pn - 1) / Pn;

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

        std::vector<double> A(k_ceil * m_ceil, 0);
        
        if (col == 0) { // Generuję kawałek macierzy A
            for (int i = 0; i < dim_m; i++) {
                for (int j = 0; j < dim_k; j++) {  
                    A[i * k_ceil + j] = generate_double(sA, i + offset_m, j + offset_k);
                    //std::cout << A[i * dim_k + j] << ", ";
                }
                //std::cout << "\n";
            }
        }

        MPI_Bcast(
            &A[0],  /* the message will be written here */
                    /* if my_rank==root, the message will be read from here */
            m_ceil * k_ceil,  /* number of items in the message */
            MPI_DOUBLE, /* type of data in the message */
            0,   /* if my_rank==root, I'm sending, otherwise I'm receiving */
            row_slice_comm  /* communicator to use */
        );

        // if (col == Pn - 1) {
        //     for (int i = 0; i < dim_m; i++) {
        //         for (int j = 0; j < dim_k; j++) {  
        //             std::cout << A[i * k_ceil + j] << ", ";
        //         }
        //         std::cout << "\n";
        //     }
        // }

        MPI_Comm col_slice_comm; // ten sam col w tym samym slice
        MPI_Comm_split(active, col * Pk + slice, row, &col_slice_comm);

        std::vector<double> B(k_ceil * n_ceil, 0);

        if (row == 0) { // Generuję kawałek macierzy B
            for (int i = 0; i < dim_n; i++) {
                for (int j = 0; j < dim_k; j++) {
                    B[i * k_ceil + j] = generate_double(sB, j + offset_k, i + offset_n);
                    // std::cout << B[i * dim_k + j] << " ";
                }
            }
        }

        MPI_Bcast(
            &B[0],  /* the message will be written here */
                    /* if my_rank==root, the message will be read from here */
            n_ceil * k_ceil,  /* number of items in the message */
            MPI_DOUBLE, /* type of data in the message */
            0,   /* if my_rank==root, I'm sending, otherwise I'm receiving */
            col_slice_comm  /* communicator to use */
        );

        // if (row == Pm - 1) {
        //     for (int i = 0; i < dim_n; i++) {
        //         for (int j = 0; j < dim_k; j++) {  
        //             std::cout << B[i * k_ceil + j] << ", ";
        //         }
        //     }
        // }        

        std::vector<double> C(m_ceil * n_ceil, 0);

        for (int i = 0; i < dim_m; i++) {
            for (int j = 0; j < dim_n; j++) {
                for (int x = 0; x < dim_k; x++) {
                    C[i * n_ceil + j] += A[i * k_ceil + x] * B[j * k_ceil + x];
                }
            }
        }

        MPI_Comm row_col_comm; // ten sam row i col
        MPI_Comm_split(active, row * Pn + col, slice, &row_col_comm);

        std::vector<double> R(m_ceil * n_ceil, 0);

        // for (int i = 0; i < dim_m; i++) {
        //     for (int j = 0; j < dim_n; j++) {        
        //         MPI_Reduce(&C[i * n_ceil + j], &R[i * n_ceil + j], 1, MPI_DOUBLE, MPI_SUM, 0,
        //         row_col_comm);
        //     }
        // }

        MPI_Reduce(&C[0], &R[0], n_ceil * m_ceil, MPI_DOUBLE, MPI_SUM, 0,
                row_col_comm);

        // if (rank == 0) {
        //     std::cout << slice << " " << row << " " << col << "?\n";
        //     std::cout << dim_k << " " << dim_m << " " << dim_n << "?\n";
        // }

        // if (slice == 0 && row == 0 && col == 0) {
        //     for (int i = 0; i < dim_m; i++) {
        //         for (int j = 0; j < dim_n; j++) {
        //             std::cout << R[i * n_ceil + j] << " ";
        //         }
        //         std::cout << "\n";
        //     }
        // }

        // if (slice == 0 && col == 0) std::cout << rank << ", ";

        // if (slice == 0) {

        //     std::vector<double> buf(n_ceil * Pn, 0);
        //     for (int i = 0; i < m_ceil; i++) {
        //         MPI_Gather(
        //             &R[i * n_ceil],
        //             n_ceil,
        //             MPI_DOUBLE,
        //             &buf[0],
        //             n_ceil,
        //             MPI_DOUBLE,
        //             0,
        //             row_slice_comm
        //         );

        //         if (row == 0 && col == 0) {

        //             for (int proc_col = 0; proc_col < Pn; proc_col++) {
        //                 for (int i = proc_col * n_ceil; i < proc_col * n_ceil + (proc_col < n % Pn ? n_ceil : (n / Pn)); i++) {
        //                     std::cout << buf[i] << " ";
        //                 }
        //             }
        //             std::cout << "\n";

        //             // Z zerami
        //             // for (int i = 0; i < n_ceil * Pn; i++) {
        //             //     std::cout << buf[i] << " ";
        //             // }
        //             // std::cout << "\n";

        //             for (int proc_row = 1; proc_row < Pm; proc_row++) {
        //                 int proc_rank = proc_row * Pn;
        //                 // std::cout << proc_rank << ", ";
        //                 MPI_Recv(
        //                     &buf[0], /* where the message will be saved */
        //                     n_ceil * Pn, /* max number of elements we expect */
        //                     MPI_DOUBLE, /* type of data in the message */
        //                     proc_rank, /* if not MPI_ANY_SOURCE, receive only from source with the given rank  */
        //                     MPI_ANY_TAG, /* if not MPI_ANY_TAG, receive only with a certain tag */
        //                     active, /* communicator to use */
        //                     MPI_STATUS_IGNORE /* if not MPI_STATUS_IGNORE, write comm info here */
        //                 );

        //                 for (int proc_col = 0; proc_col < Pn; proc_col++) {
        //                     for (int i = proc_col * n_ceil; i < proc_col * n_ceil + (proc_col < n % Pn ? n_ceil : (n / Pn)); i++) {
        //                         std::cout << buf[i] << " ";
        //                     }
        //                 }
        //                 std::cout << "\n";

        //                 // Z zerami
        //                 // for (int i = 0; i < n_ceil * Pn; i++) {
        //                 //     std::cout << buf[i] << " ";
        //                 // }
        //                 // std::cout << "\n";
        //             }                 
        //         }
                
        //         else if (col == 0) {
        //             // std::cout << rank;
        //             MPI_Send(
        //                 &buf[0],  /* pointer to the message */
        //                 n_ceil * Pn, /* number of items in the message */
        //                 MPI_DOUBLE, /* type of data in the message */
        //                 0, /* rank of the destination process */
        //                 0, /* app-defined message type */
        //                 active /* communicator to use */
        //             );
        //         }

        //     }
        // }

        if (slice == 0) {
            int next_row = 0;
            int print = 0;

            while (next_row < dim_m) {
                if (col == 0 && row == 0 && next_row == 0) {
                    //std::cout << "zaczynam\n";
                }
                else {
                    MPI_Recv(
                        &print,
                        1,
                        MPI_INTEGER,
                        MPI_ANY_SOURCE,
                        MPI_ANY_TAG, /* if not MPI_ANY_TAG, receive only with a certain tag */
                        active, /* communicator to use */
                        MPI_STATUS_IGNORE /* if not MPI_STATUS_IGNORE, write comm info here */
                    );
                    //std::cout << "dostaje\n";                    
                }
                if (Pn == 1) {
                    while (next_row < dim_m) {
                        for (int i = 0; i < dim_n; i++) {
                            std::cout << R[next_row * n_ceil + i] << " ";
                        }
                        next_row++; 
                        std::cout << "\n";             
                    }
                    if (rank + Pn < Pn * Pm) {
                        MPI_Send(
                            &print,
                            1,
                            MPI_INTEGER,
                            rank + Pn,
                            0,
                            active
                        );                        
                    }
                }
                else {
                    for (int i = 0; i < dim_n; i++) {
                        std::cout << R[next_row * n_ceil + i] << " ";
                    }
                    next_row++;

                    if (col == Pn - 1) std::cout << "\n";

                    int dest;
                    // if (rank + 1 < (rank / Pn + 1) * Pn) {
                    //     dest = rank + 1;
                    // }
                    // else {
                    //     dest = (rank / Pn) * Pn;
                    // }
                    if (rank + 1 < (rank / Pn + 1) * Pn || next_row == dim_m) {
                        dest = rank + 1;
                    }
                    else {
                        dest = (rank / Pn) * Pn;
                    }

                    //std::cout << "wysylam\n";
                    if (dest < Pn * Pm)
                    MPI_Send(
                        &print,
                        1,
                        MPI_INTEGER,
                        dest,
                        0,
                        active
                    );
                    //std::cout << "wyslane\n";
                }
            }

        }

    } 


    MPI_Finalize();
    return 0;
    
}