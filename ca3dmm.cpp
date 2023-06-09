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

        MPI_Comm slice_comm;
        MPI_Comm_split(active, slice, row * Pn + col, &slice_comm);

        int s = std::min(Pm, Pn);

        int row_in_group = row % s;
        int col_in_group = col % s;
        int group = (row * Pn + col) / (s * s);

        MPI_Comm equi_comm; // Odpowiadające sobie procesy w różnych grupach
        MPI_Comm_split(slice_comm, row_in_group * s + col_in_group, group, &equi_comm);

        if (Pn <= Pm) { // col == col_in_group
            // Pierwsza grupa generuje całą macierz Bi i rozsyła swoim odpowiednikom
            int ceil_n = (n + s - 1) / s;
            int ceil_k = (dim_k + s - 1) / s;

            std::vector<double> B(ceil_n * ceil_k, 0);

            int part_n = col < n % Pn ? (n + Pn - 1) / Pn : n / Pn;
            if (group == 0) {
                // int part_n = col_in_group < n % s ? (n + s - 1) / s : n / s;
                int part_k = row_in_group < dim_k % s ? (dim_k + s - 1) / s : dim_k / s;

                int offset_n = col_in_group * (n / Pn) + std::min(col_in_group, n % Pn);
                int offset_k = slice * (k / Pk) + std::min(slice, k % Pk) + row_in_group * (dim_k / s) + std::min(row_in_group, dim_k % s);


                for (int i = 0; i < part_n; i++) {
                    for (int j = 0; j < part_k; j++) {
                        B[i * ceil_k + j] = generate_double(sB, j + offset_k, i + offset_n);
                    }
                }
            }

            MPI_Bcast(
                &B[0],  /* the message will be written here */
                        /* if my_rank==root, the message will be read from here */
                ceil_n * ceil_k,  /* number of items in the message */
                MPI_DOUBLE, /* type of data in the message */
                0,   /* if my_rank==root, I'm sending, otherwise I'm receiving */
                equi_comm  /* communicator to use */
            );


            // Każdy proces generuje swój kawałeczek macierzy Ai

            int ceil_m = (m + Pm - 1) / Pm;
            std::vector<double> A(ceil_m * ceil_k, 0);

            int part_m = row < m % Pm ? (m + Pm - 1) / Pm : m / Pm;
            int part_k = col < dim_k % Pn ? (dim_k + Pn - 1) / Pn : dim_k / Pn;

            int offset_m = row * (m / Pm) + std::min(row, m % Pm);
            int offset_k = slice * (k / Pk) + std::min(slice, k % Pk) + col * (dim_k / s) + std::min(col, dim_k % s);

            for (int i = 0; i < part_m; i++) {
                for (int j = 0; j < part_k; j++) {
                    A[i * ceil_k + j] = generate_double(sA, i + offset_m, j + offset_k);
                }
            }

            // Robimy shift

            MPI_Comm group_comm;
            MPI_Comm_split(slice_comm, group, row_in_group * s + col_in_group, &group_comm);

            MPI_Comm col_comm;
            MPI_Comm_split(group_comm, col_in_group, row_in_group, &col_comm);

            MPI_Comm row_comm;
            MPI_Comm_split(group_comm, row_in_group, col_in_group, &row_comm);

            // if (group == 0 && slice == 0) {
            //     std::cout << rank << " (" << row_in_group << ", " << col_in_group << ")\n";
            // }

            std::vector<double> bufB(ceil_n * ceil_k, 0);
            std::vector<double> bufA(ceil_m * ceil_k, 0);

            // Przesuwamy kolumny B
            if (col_in_group > 0) {
                if (row_in_group == 0) {
                    MPI_Recv(
                        &bufB[0],
                        ceil_n * ceil_k,
                        MPI_DOUBLE,
                        (row_in_group + col_in_group) % s,
                        MPI_ANY_TAG,
                        col_comm,
                        MPI_STATUS_IGNORE
                    );
                    MPI_Send(
                        &B[0],
                        ceil_n * ceil_k,
                        MPI_DOUBLE,
                        row_in_group - col_in_group < 0 ? row_in_group - col_in_group + s : row_in_group - col_in_group,
                        0,
                        col_comm
                    );
                    swap(bufB, B);
                }
                else {
                    MPI_Send(
                        &B[0],
                        ceil_n * ceil_k,
                        MPI_DOUBLE,
                        row_in_group - col_in_group < 0 ? row_in_group - col_in_group + s : row_in_group - col_in_group,
                        0,
                        col_comm
                    );
                    MPI_Recv(
                        &B[0],
                        ceil_n * ceil_k,
                        MPI_DOUBLE,
                        (row_in_group + col_in_group) % s,
                        MPI_ANY_TAG,
                        col_comm,
                        MPI_STATUS_IGNORE
                    );
                }
            }

            // Przesuwamy wiersze A
            if (row_in_group > 0) {
                if (col_in_group == 0) {
                    MPI_Recv(
                        &bufA[0],
                        ceil_m * ceil_k,
                        MPI_DOUBLE,
                        (col_in_group + row_in_group) % s,
                        MPI_ANY_TAG,
                        row_comm,
                        MPI_STATUS_IGNORE
                    );
                    MPI_Send(
                        &A[0],
                        ceil_m * ceil_k,
                        MPI_DOUBLE,
                        col_in_group - row_in_group < 0 ? col_in_group - row_in_group + s : col_in_group - row_in_group,
                        0,
                        row_comm
                    );
                    swap(bufA, A);
                }
                else {
                    MPI_Send(
                        &A[0],
                        ceil_m * ceil_k,
                        MPI_DOUBLE,
                        col_in_group - row_in_group < 0 ? col_in_group - row_in_group + s : col_in_group - row_in_group,
                        0,
                        row_comm
                    );
                    MPI_Recv(
                        &A[0],
                        ceil_m * ceil_k,
                        MPI_DOUBLE,
                        (col_in_group + row_in_group) % s,
                        MPI_ANY_TAG,
                        row_comm,
                        MPI_STATUS_IGNORE
                    );
                }
            }

            std::vector<double> C(ceil_n * ceil_m, 0);

            for (int step = 0; step < s; step++) {

                // if (rank == 3) {
                //     std::cout<<"step "<<step<<" out of "<<s<<"\n";
                //     std::cout << "multiplying\n";
                //     for (int i = 0; i < ceil_m; i++) {
                //         for (int j = 0; j < ceil_k; j++) {
                //             std::cout <<A[i * ceil_k + j] << " ";
                //         }
                //         std::cout << "\n";
                //     }
                //     std::cout << "with\n";
                //     for (int i = 0; i < ceil_k; i++) {
                //         for (int j = 0; j < ceil_n; j++) {
                //             std::cout <<B[i * ceil_n + j] << " ";
                //         }
                //         std::cout << "\n";
                //     }
                // }

                for (int i = 0; i < ceil_m; i++) {
                    for (int j = 0; j < ceil_n; j++) {
                        for (int x = 0; x < ceil_k; x++) {
                            C[i * ceil_n + j] += A[i * ceil_k + x] * B[j * ceil_k + x];
                        }
                    }
                }

                if (step + 1 < s) {
                    // Przesuwamy kolumny B
                    if (row_in_group == 0) {
                        MPI_Recv(
                            &bufB[0],
                            ceil_n * ceil_k,
                            MPI_DOUBLE,
                            (row_in_group + 1) % s,
                            MPI_ANY_TAG,
                            col_comm,
                            MPI_STATUS_IGNORE
                        );
                        MPI_Send(
                            &B[0],
                            ceil_n * ceil_k,
                            MPI_DOUBLE,
                            row_in_group - 1 < 0 ? row_in_group - 1 + s : row_in_group - 1,
                            0,
                            col_comm
                        );
                        swap(bufB, B);
                    }
                    else {
                        MPI_Send(
                            &B[0],
                            ceil_n * ceil_k,
                            MPI_DOUBLE,
                            row_in_group - 1 < 0 ? row_in_group - 1 + s : row_in_group - 1,
                            0,
                            col_comm
                        );  
                        MPI_Recv(
                            &B[0],
                            ceil_n * ceil_k,
                            MPI_DOUBLE,
                            (row_in_group + 1) % s,
                            MPI_ANY_TAG,
                            col_comm,
                            MPI_STATUS_IGNORE
                        );
                    }
                    // Przesuwamy wiersze A
                    if (col_in_group == 0) {
                        MPI_Recv(
                            &bufA[0],
                            ceil_m * ceil_k,
                            MPI_DOUBLE,
                            (col_in_group + 1) % s,
                            MPI_ANY_TAG,
                            row_comm,
                            MPI_STATUS_IGNORE
                        );
                        MPI_Send(
                            &A[0],
                            ceil_m * ceil_k,
                            MPI_DOUBLE,
                            col_in_group - 1 < 0 ? col_in_group - 1 + s : col_in_group - 1,
                            0,
                            row_comm
                        );
                        swap(bufA, A);
                    }
                    else {
                        MPI_Send(
                            &A[0],
                            ceil_m * ceil_k,
                            MPI_DOUBLE,
                            col_in_group - 1 < 0 ? col_in_group - 1 + s : col_in_group - 1,
                            0,
                            row_comm
                        );           
                        MPI_Recv(
                            &A[0],
                            ceil_m * ceil_k,
                            MPI_DOUBLE,
                            (col_in_group + 1) % s,
                            MPI_ANY_TAG,
                            row_comm,
                            MPI_STATUS_IGNORE
                        );
                    }
                }
            }

            MPI_Comm row_col_comm; // ten sam row i col
            MPI_Comm_split(active, row * Pn + col, slice, &row_col_comm);

            std::vector<double> R(ceil_m * ceil_n, 0);

            MPI_Reduce(&C[0], &R[0], ceil_n * ceil_m, MPI_DOUBLE, MPI_SUM, 0,
                    row_col_comm);

            // if (slice == 0 && row == 0 && col == 0) {
            //     for (int i = 0; i < ceil_m; i++) {
            //         for (int j = 0; j < ceil_n; j++) {
            //             std::cout << C[i * ceil_n + j] << " ";
            //         }
            //         std::cout << "\n";
            //     }
            // }

            if (slice == 0) {
                int next_row = 0;
                int print = 0;

                while (next_row < part_m) {
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
                        while (next_row < part_m) {
                            for (int i = 0; i < part_n; i++) {
                                std::cout << R[next_row * ceil_n + i] << " ";
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
                        for (int i = 0; i < part_n; i++) {
                            std::cout << R[next_row * ceil_n + i] << " ";
                        }
                        next_row++;

                        if (col == Pn - 1) std::cout << "\n";

                        int dest;
                        if (rank + 1 < (rank / Pn + 1) * Pn || next_row == part_m) {
                            dest = rank + 1;
                        }
                        else {
                            dest = (rank / Pn) * Pn;
                        }

                        //std::cout << "wysylam\n";
                        if (dest < Pn * Pm) {
                            MPI_Send(
                                &print,
                                1,
                                MPI_INTEGER,
                                dest,
                                0,
                                active
                            );
                        }

                        //std::cout << "wyslane\n";
                    }
                }

            }
        }

    }


    MPI_Finalize();
    return 0;

}
