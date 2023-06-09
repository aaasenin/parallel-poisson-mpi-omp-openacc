#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

const double A1 = 0;
const double A2 = 2;
const double B1 = 0;
const double B2 = 1;

const double EPSILON = 1e-6;




# pragma acc routine
double dudx(double x, double y) { return -sin(M_PI * x * y) * M_PI * y; }

# pragma acc routine
double dudy(double x, double y) { return -sin(M_PI * x * y) * M_PI * x; }

# pragma acc routine
double u(double x, double y){ return 1 + cos(M_PI * x * y); }

# pragma acc routine
double k(double x, double y){ return 4 + x + y; }

# pragma acc routine
double F(double x, double y){
    return sin(M_PI * x * y) * M_PI * (x + y) + 
    k(x, y) * cos(M_PI * x * y) * M_PI * M_PI * (x * x + y * y);
}

# pragma acc routine
double psi_R(double x, double y){
    return u(x, y) + k(x, y) * dudx(x, y);
}

# pragma acc routine
double psi_L(double x, double y){
    return u(x, y) - k(x, y) * dudx(x, y);
}

# pragma acc routine
double psi_T(double x, double y){
    return k(x, y) * dudy(x, y);
}

# pragma acc routine
double psi_B(double x, double y){
    return -k(x, y) * dudy(x, y);
}

# pragma acc routine
double rho(int i, int j, int i_max, int j_max, 
           int left_flag, int right_flag, 
           int top_flag, int bottom_flag) {
    double result = 1;
    if ((i == 1 && left_flag) || (i == i_max && right_flag)) {
        result *= 0.5;
    }
    if ((j == 1 && bottom_flag) || (j == j_max && top_flag)) {
        result *= 0.5;
    }    
    return result;
}

void print_matrix(int M, int N, double (*Matrix)[N + 2], char* title) {
    int curr_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);
    printf("%s\n", title);
    for (int i = 0; i <= M + 1; i++) {
        for (int j = 0; j <= N + 1; j++) {
            printf("%3.8f(rank %d) ", Matrix[i][j], curr_rank);
        }
        printf("\n");
    }
}

double dot_product(int M, int N, double (*restrict X)[N + 2], double (*restrict Y)[N + 2], double h1, double h2,
                   int left_flag, int right_flag, 
                   int top_flag, int bottom_flag) {
    double result = 0.;

    #pragma acc data present(X[:M + 2][:N + 2], Y[:M + 2][:N + 2])
    {
    # pragma acc parallel loop independent reduction(+:result)
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            result += X[i][j] * Y[i][j] * rho(i, j, M, N, left_flag, right_flag, top_flag, bottom_flag) * h1 * h2;
        }
    }
    }
    return result;
}

double norm(int M, int N, double (*X)[N + 1], double h1, double h2,
            int left_flag, int right_flag, 
            int top_flag, int bottom_flag) {
    return dot_product(M, N, X, X, h1, h2, left_flag, right_flag, top_flag, bottom_flag);
}

void init_B_fill_border(int M, int N, double (*Matrix)[N + 2], double h1, double h2, 
                   double x_min, double y_min, int left_flag, int right_flag,
                   int top_flag, int bottom_flag) {
    int i, j;
    double x, y; double psi_ij;
    int i_max = M;
    int j_max = N;

    /* left */
    if (left_flag) {
        i = 0;
        x = x_min + i * h1;

        #pragma omp parallel for private(j, y)
        for (j = 0; j < j_max; j++) {
            y = y_min + j * h2;
            Matrix[i + 1][j + 1] = F(x, y) + psi_L(x, y) * 2 / h1;
        }
    }

    /* right */
    if (right_flag) {
        i = i_max - 1;
        x = x_min + i * h1;

        #pragma omp parallel for private(j, y)
        for (j = 0; j < j_max; j++) {
            y = y_min + j * h2;
            Matrix[i + 1][j + 1] = F(x, y) + psi_R(x, y) * 2 / h1;
        }
    }

    /* bottom */
    if (bottom_flag) {
        j = 0;
        y = y_min + j * h2;

        #pragma omp parallel for private(i, x)
        for (i = 0; i < i_max; i++) {
            x = x_min + i * h1;
            Matrix[i + 1][j + 1] = F(x, y) + psi_B(x, y) * 2 / h2;
        }
    }

    /* top */
    if (top_flag) {
        j = j_max - 1;
        y = y_min + j * h2;

        #pragma omp parallel for private(i, x)
        for (i = 0; i < i_max; i++) {
            x = x_min + i * h1;
            Matrix[i + 1][j + 1] = F(x, y) + psi_T(x, y) * 2 / h2;
        }    
    }

    /* left bottom */
    if (left_flag && bottom_flag) {
        i = 0; j = 0;
        x = x_min + i * h1;
        y = y_min + j * h2;
        psi_ij = (psi_B(x, y) + psi_L(x, y)) / 2;
        Matrix[i + 1][j + 1] = F(x, y) + psi_ij * (2 / h1 + 2 / h2);
    }
    
    /* left top */
    if (left_flag && top_flag) {
        i = 0; j = j_max - 1;
        x = x_min + i * h1;
        y = y_min + j * h2;
        psi_ij = (psi_T(x, y) + psi_L(x, y)) / 2;
        Matrix[i + 1][j + 1] = F(x, y) + psi_ij * (2 / h1 + 2 / h2);
    }

    /* right top */
    if (right_flag && top_flag) {
        i = i_max - 1; j = j_max - 1;
        x = x_min + i * h1;
        y = y_min + j * h2;
        psi_ij = (psi_T(x, y) + psi_R(x, y)) / 2;
        Matrix[i + 1][j + 1] = F(x, y) + psi_ij * (2 / h1 + 2 / h2);
    }

    /* right bottom */
    if (right_flag && bottom_flag) {
        i = i_max - 1; j = 0;
        x = x_min + i * h1;
        y = y_min + j * h2;
        psi_ij = (psi_B(x, y) + psi_R(x, y)) / 2;
        Matrix[i + 1][j + 1] = F(x, y) + psi_ij * (2 / h1 + 2 / h2);
    }
}

void init_B(int M, int N, double (*B)[N + 2], double h1, double h2, double x_min, double y_min) {
    int i, j;
    double x, y;
    int i_max = M;
    int j_max = N;

    #pragma omp parallel for private(i, j, x, y)
    for (i = 0; i <= i_max + 1; i++) {
        for (j = 0; j <= j_max + 1; j++) {
            x = x_min + (i - 1) * h1;
            y = y_min + (j - 1) * h2;
            B[i][j] = F(x, y);
        }
    }
}

void A_mul_vec_fill_border(int M, int N, double (*restrict w)[N + 2], double (*restrict res)[N + 2], 
                           double h1, double h2, double x_min, double y_min, 
                           int i_max, int j_max, int left_flag, int right_flag,
                           int top_flag, int bottom_flag) {
    #pragma acc data present(w[:M + 2][:N + 2], res[:M + 2][:N + 2])
    {


    /* left */
    if (left_flag) {
        #pragma acc parallel loop
        for (int j = 1; j <= j_max; j++) {
            double aw, bw, x, y;
            y = y_min + (j - 1) * h2;

            x = x_min + h1;
            aw =  k(x - 0.5 * h1, y) * (w[2][j] - w[1][j]) / h1;

            x = x_min;
            bw = 1 / h2 * (k(x, y + 0.5 * h2) * (w[1][j + 1] - w[1][j]) / h2 - \
                        k(x, y - 0.5 * h2) * (w[1][j] - w[1][j - 1]) / h2);

            res[1][j] = -2/h1 * aw - bw + (2 / h1) * w[1][j];
        }
    }

    /* right */
    if (right_flag) {
        #pragma acc parallel loop
        for (int j = 1; j <= j_max; j++) {
            double aw, bw, x, y;
            y = y_min + (j - 1) * h2;

            x = x_min + (i_max - 1) * h1;
            aw = k(x - 0.5 * h1, y) * (w[i_max][j] - w[i_max - 1][j]) / h1;

            bw = 1 / h2 * (k(x, y + 0.5 * h2) * (w[i_max][j + 1] - w[i_max][j]) / h2 - \
                        k(x, y - 0.5 * h2) * (w[i_max][j] - w[i_max][j - 1]) / h2);

            res[i_max][j] = 2/h1 * aw - bw + (2 / h1) * w[i_max][j];
        }
    }

    /* bottom */
    if (bottom_flag) {
        #pragma acc parallel loop
        for (int i = 1; i <= i_max; i++) {
            double aw, bw, x, y;
            x = x_min + (i - 1) * h1;

            y = y_min;
            aw = 1 / h1 * (k(x + 0.5 * h1, y) * (w[i + 1][1] - w[i][1]) / h1 - \
                            k(x - 0.5 * h1, y) * (w[i][1] - w[i - 1][1]) / h1);

            y = y_min + h2;                
            bw = k(x, y - 0.5 * h2) * (w[i][2] - w[i][1]) / h2;

            res[i][1] = -2/h2 * bw - aw;
        }
    }

    /* top */
    if (top_flag) {
        #pragma acc parallel loop
        for (int i = 1; i <= i_max; i++) {
            double aw, bw, x, y;
            x = x_min + (i - 1) * h1;

            y = y_min + (j_max - 1) * h2;
            aw = 1 / h1 * (k(x + 0.5 * h1, y) * (w[i + 1][j_max] - w[i][j_max]) / h1 - \
                            k(x - 0.5 * h1, y) * (w[i][j_max] - w[i - 1][j_max]) / h1);
            bw = k(x, y - 0.5 * h2) * (w[i][j_max] - w[i][j_max - 1]) / h2;

            res[i][j_max] = 2/h2 * bw - aw;
        }
    }
        
    
        
    double aw, bw, x, y;

    /* left bottom */
    if (left_flag && bottom_flag) {
        x = x_min + h1; y = y_min;
        aw = k(x - 0.5 * h1, y) * (w[2][1] - w[1][1]) / h1;
        x = x_min; y = y_min + h2;
        bw = k(x, y - 0.5 * h2) * (w[1][2] - w[1][1]) / h2;
        res[1][1] = -2/h1 * aw - 2/h2 * bw + (2 / h1) * w[1][1];
    }
    
    /* left top */
    if (left_flag && top_flag) {
        x = x_min + h1; y = y_min + (j_max - 1) * h2;
        aw = k(x - 0.5 * h1, y) * (w[2][j_max] - w[1][j_max]) / h1;
        x = x_min; y = y_min + (j_max - 1) * h2;
        bw = k(x, y - 0.5 * h2) * (w[1][j_max] - w[1][j_max - 1]) / h2;
        res[1][j_max] = -2/h1 * aw + 2/h2 * bw + (2 / h1) * w[1][j_max];
    }

    /* right top */
    if (right_flag && top_flag) {
        x = x_min + (i_max - 1) * h1; y = y_min + (j_max - 1) * h2;
        aw = k(x - 0.5 * h1, y) * (w[i_max][j_max] - w[i_max - 1][j_max]) / h1;
        bw = k(x, y - 0.5 * h2) * (w[i_max][j_max] - w[i_max][j_max - 1]) / h2;
        res[i_max][j_max] = 2/h1 * aw + 2/h2 * bw + (2 / h1) * w[i_max][j_max];
    }

    /* right bottom */
    if (right_flag && bottom_flag) {
        x = x_min + (i_max - 1) * h1; y = y_min;
        aw = k(x - 0.5 * h1, y) * (w[i_max][1] - w[i_max - 1][1]) / h1;
        x = x_min + (i_max - 1) * h1; y = y_min + h2;
        bw = k(x, y - 0.5 * h2) * (w[i_max][2] - w[i_max][1]) / h2;
        res[i_max][1] = 2/h1 * aw - 2/h2 * bw + (2 / h1) * w[i_max][1];
    }
    }
}

void A_mul_vec(int M, int N, double (*restrict w)[N + 2], double (*restrict res)[N + 2], double h1, double h2, double x_min, double y_min) {
    
    #pragma acc data present(w[:M + 2][:N + 2], res[:M + 2][:N + 2])
    {
    # pragma acc parallel 
    {
    # pragma acc loop independent
    for (int i = 0; i <= M + 1; i++) {
        res[i][0] = w[i][0];
        res[i][N + 1] = w[i][N + 1];
    }
    
    # pragma acc loop independent
    for (int j = 0; j <= N + 1; j++) {
        res[0][j] = w[0][j];
        res[M + 1][j] = w[M + 1][j];
    }

    # pragma acc loop independent
    for (int i = 1; i <= M; i++) {
        # pragma acc loop independent 
        for (int j = 1; j <= N; j++) {
                double x = x_min + (i - 1) * h1;
                double y = y_min + (j - 1) * h2;
                double aw = 1 / h1 * (k(x + 0.5 * h1, y) * (w[i + 1][j] - w[i][j]) / h1 - \
                                k(x - 0.5 * h1, y) * (w[i][j] - w[i - 1][j]) / h1);
                double bw = 1 / h2 * (k(x, y + 0.5 * h2) * (w[i][j + 1] - w[i][j]) / h2 - \
                        k(x, y - 0.5 * h2) * (w[i][j] - w[i][j - 1]) / h2);

                res[i][j] = -aw - bw;
        }
    }
    }
    }
    
}

void sync_borders(int x_n, int y_n, MPI_Comm MPI_COMM_CARTESIAN_TOPOLOGY, int left_flag, int right_flag, 
                  int top_flag, int bottom_flag, int process_dims[2], int curr_coords[2],
                  double left_send_buf[y_n], double right_send_buf[y_n], double top_send_buf[x_n], double bottom_send_buf[x_n],
                  double left_recv_buf[y_n], double right_recv_buf[y_n], double top_recv_buf[x_n], double bottom_recv_buf[x_n],
                  double (*w)[y_n + 2], int MPI_tag, double x_min, double y_min, double h1, double h2, int i_x, int i_y) {
    int near_coords[2];
    int near_rank; int i, j;

    MPI_Status status; MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};;

    // left send
    if ((left_flag == 0) && (process_dims[0] > 1)) {
        near_coords[0] = curr_coords[0] - 1;
        near_coords[1] = curr_coords[1];

        #pragma acc data present(w[:x_n + 2][y_n + 2], left_send_buf[:y_n])
        {
        for (j = 0; j < y_n; j++) {
            left_send_buf[j] = w[1][j + 1];
        }
        }

        # pragma acc data copyout(left_send_buf[:y_n])
        {
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Isend(left_send_buf, y_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &requests[0]);
        }
    }

    // right send
    if ((right_flag == 0) && (process_dims[0] > 1)) {
        near_coords[0] = curr_coords[0] + 1;
        near_coords[1] = curr_coords[1];
        
        #pragma acc data present(w[:x_n + 2][y_n + 2], right_send_buf[:y_n])
        {
        for (j = 0; j < y_n; j++) {
            right_send_buf[j] = w[x_n][j + 1];
        }
        }

        # pragma acc data copyout(right_send_buf[:y_n])
        {
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Isend(right_send_buf, y_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &requests[1]);
        }
    }

    // top send
    if ((top_flag == 0) && (process_dims[1] > 1)) {
        near_coords[0] = curr_coords[0];
        near_coords[1] = curr_coords[1] + 1;

        #pragma acc data present(w[:x_n + 2][y_n + 2], top_send_buf[:x_n])
        {
        for (i = 0; i < x_n; i++) {
            top_send_buf[i] = w[i + 1][y_n];
        }
        }

        # pragma acc data copyout(top_send_buf[:x_n])
        {
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Isend(top_send_buf, x_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &requests[2]);
        }
    }

    // bottom send
    if ((bottom_flag == 0) && (process_dims[1] > 1)) {
        near_coords[0] = curr_coords[0];
        near_coords[1] = curr_coords[1] - 1;

        #pragma acc data present(w[:x_n + 2][y_n + 2], bottom_send_buf[:x_n])
        {
        for (i = 0; i < x_n; i++) {
            bottom_send_buf[i] = w[i + 1][1];
        }
        }

        # pragma acc data copyout(bottom_send_buf[:x_n])
        {
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Isend(bottom_send_buf, x_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &requests[3]);
        }
    }

    // left recieve
    if ((left_flag && (process_dims[0] > 1)) || (process_dims[0] == 1)) {

        #pragma acc data present(w[:x_n + 2][y_n + 2])
        {
        for (j = 0; j < y_n; j++) {
            w[0][j + 1] = u(x_min + (i_x - 1) * h1, y_min + (i_y + j) * h2);
        }
        }
    }
    else {
        near_coords[0] = curr_coords[0] - 1;
        near_coords[1] = curr_coords[1];
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Recv(left_recv_buf, y_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &status);

        #pragma acc data present(w[:x_n + 2][y_n + 2], left_send_buf[:y_n])
        {
        for (j = 0; j < y_n; j++) {
            w[0][j + 1] = left_recv_buf[j];
        }
        }
    }

    // right recieve
    if ((right_flag && (process_dims[0] > 1)) || (process_dims[0] == 1)) {

        #pragma acc data present(w[:x_n + 2][y_n + 2])
        {
        for (j = 0; j < y_n; j++) {
            w[x_n + 1][j + 1] = u(x_min + (i_x + x_n) * h1, y_min + (i_y + j) * h2);
        }
        }
    }
    else {
        near_coords[0] = curr_coords[0] + 1;
        near_coords[1] = curr_coords[1];
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Recv(right_recv_buf, y_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &status);

        #pragma acc data present(w[:x_n + 2][y_n + 2], right_recv_buf[:y_n])
        {
        for (j = 0; j < y_n; j++) {
            w[x_n + 1][j + 1] = right_recv_buf[j];
        }
        }
    }

    // top recieve
    if ((top_flag && (process_dims[1] > 1)) || (process_dims[1] == 1)) {

        #pragma acc data present(w[:x_n + 2][y_n + 2])
        {
        for (i = 0; i < x_n; i++) {
            w[i + 1][y_n + 1] = u(x_min + (i_x + i) * h1, y_min + (i_y + y_n) * h2);
        }
        }
    }
    else {
        near_coords[0] = curr_coords[0];
        near_coords[1] = curr_coords[1] + 1;
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Recv(top_recv_buf, x_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &status);

        #pragma acc data present(w[:x_n + 2][y_n + 2], top_recv_buf[:x_n])
        {
        for (i = 0; i < x_n; i++) {
            w[i + 1][y_n + 1] = top_recv_buf[i];
        }
        }
    }

    // bottom recieve
    if ((bottom_flag && (process_dims[1] > 1)) || (process_dims[1] == 1)) {

        #pragma acc data present(w[:x_n + 2][y_n + 2])
        {
        for (i = 0; i < x_n; i++) {
            w[i + 1][0] = u(x_min + (i_x + i) * h1, y_min + (i_y - 1) * h2);
        }
        }
    }
    else {
        near_coords[0] = curr_coords[0];
        near_coords[1] = curr_coords[1] - 1;
        MPI_Cart_rank(MPI_COMM_CARTESIAN_TOPOLOGY, near_coords, &near_rank);
        MPI_Recv(bottom_recv_buf, x_n, MPI_DOUBLE, near_rank, MPI_tag, MPI_COMM_CARTESIAN_TOPOLOGY, &status);

        #pragma acc data present(w[:x_n + 2][y_n + 2], bottom_recv_buf[:x_n])
        {
        for (i = 0; i < x_n; i++) {
            w[i + 1][0] = bottom_recv_buf[i];
        }
        }
    }

    // MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    for (i = 0; i <= 3; i++) {
       MPI_Wait(&requests[i], &status);
    }
}

double max_abs_matr(int M, int N, double (*w)[N + 2]) {
    double res = 0;

    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            double value = fabs(w[i][j]);
            if (value > res)
                res = value;
        }
    }
    return res;
}

double mean_abs_matr(int M, int N, double (*w)[N + 2]) {
    double res = 0;

    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            res += fabs(w[i][j]);
            
        }
    }
    return res / (M * N);
}

int main(int argc, char* argv[]) {
    int M = atoi(argv[argc - 2]);
    int N = atoi(argv[argc - 1]);
    // const int N = 10;
    // const int M = 10;

    double h1 = (A2 - A1) / M;
    double h2 = (B2 - B1) / N;

    printf("h1 = %3.8f h2 = %3.8f\n", h1, h2);

    MPI_Init(&argc, &argv);

    // MPI variables
    int curr_rank, n_processes; int MPI_tag = 0;
    int process_dims[2] = {0, 0};
    int grid_is_periodic[2] = {0, 0};
    int curr_coords[2];
    int left_flag = 0, right_flag = 0, top_flag = 0, bottom_flag = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    MPI_Comm MPI_COMM_CARTESIAN_TOPOLOGY;
    MPI_Dims_create(n_processes, 2, process_dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, process_dims, grid_is_periodic, 1, &MPI_COMM_CARTESIAN_TOPOLOGY);
    MPI_Cart_coords(MPI_COMM_CARTESIAN_TOPOLOGY, curr_rank, 2, curr_coords);

    if (curr_coords[0] == 0) {
        left_flag = 1;
    }
    if (curr_coords[1] == 0) {
        bottom_flag = 1;
    }
    if ((curr_coords[0] + 1) == process_dims[0]) {
        right_flag = 1;
    }
    if ((curr_coords[1] + 1) == process_dims[1]) {
        top_flag = 1;
    }

    printf("rank %d left %d right %d top %d bottom %d \n", curr_rank, left_flag, right_flag, top_flag, bottom_flag);

    int x_i, x_n, y_i, y_n;
    if ((M + 1) % process_dims[0] == 0) {
        x_n = (M + 1) / process_dims[0];
        x_i = curr_coords[0] * x_n;
    } 
    else {
        if (curr_coords[0] == 0) {
            x_n = (M + 1) / process_dims[0] + (M + 1) % process_dims[0];
            x_i = 0;
        }
        else {
            x_n = (M + 1) / process_dims[0];
            x_i = curr_coords[0] * x_n + (M + 1) % process_dims[0];
        }
    }

    if ((N + 1) % process_dims[1] == 0) {
        y_n = (N + 1) / process_dims[1];
        y_i = curr_coords[1] * y_n;
    } 
    else {
        if (curr_coords[1] == 0) {
            y_n = (N + 1) / process_dims[1] + (N + 1) % process_dims[1];
            y_i = 0;
        }
        else {
            y_n = (N + 1) / process_dims[1];
            y_i = curr_coords[1] * y_n + (N + 1) % process_dims[1];
        }
    }

    printf("rank %d x_i %d y_i %d x_n %d y_n %d \n", curr_rank, x_i, y_i, x_n, y_n);

    double *left_send_buf = malloc(sizeof(double[y_n])); double *right_send_buf = malloc(sizeof(double[y_n]));
    double *left_recv_buf = malloc(sizeof(double[y_n])); double *right_recv_buf = malloc(sizeof(double[y_n]));
    double *top_send_buf = malloc(sizeof(double[x_n])); double *bottom_send_buf = malloc(sizeof(double[x_n]));
    double *top_recv_buf = malloc(sizeof(double[x_n])); double *bottom_recv_buf = malloc(sizeof(double[x_n]));

    // algorithm variables
    int converge = 0;
    int n_iter = 0;
    int i_max = x_n; int j_max = y_n;
    double CRITERIA = 0; double criteria = 0; int MAX_ITER = 10000;
    double TAU, TAU_DOT, TAU_NORM; double tau_dot, tau_norm;

    double (*B)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*w)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*w_next)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*Aw)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*r)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*Ar)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*w_diff)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));
    double (*true_u)[y_n + 2] = malloc(sizeof(double[x_n + 2][y_n + 2]));

    double start_timestamp = MPI_Wtime();

    init_B(x_n, y_n, B, h1, h2, A1 + h1 * x_i, B1 + h2 * y_i);
    init_B_fill_border(x_n, y_n, B, h1, h2, A1 + h1 * x_i, B1 + h2 * y_i, left_flag, right_flag, top_flag, bottom_flag);

    #pragma omp parallel for
    for (int i = 0; i <= x_n + 1; i++) {
        for (int j = 0; j <= y_n; j++) {
            w[i][j] = 0;
        }
    }

    # pragma acc data copy(w[0:x_n + 2][0:y_n + 2], Aw[0:x_n + 2][0:y_n + 2], r[0:x_n + 2][0:y_n + 2], \
    B[:x_n + 2][0:y_n + 2], Ar[0:x_n + 2][0:y_n + 2], \
    left_send_buf[:y_n], right_send_buf[:y_n], left_recv_buf[:y_n], right_recv_buf[:y_n],\
    top_send_buf[:x_n], bottom_send_buf[:x_n], top_recv_buf[:x_n], bottom_recv_buf[:x_n]), \
    create(w_next[0:x_n + 2][0:y_n + 2], w_diff[0:x_n + 2][0:y_n + 2])
    {
    while (!converge && n_iter <= MAX_ITER) {

        sync_borders(x_n, y_n, MPI_COMM_CARTESIAN_TOPOLOGY, left_flag, right_flag, 
                  top_flag, bottom_flag, process_dims, curr_coords,
                  left_send_buf, right_send_buf, top_send_buf, bottom_send_buf,
                  left_recv_buf, right_recv_buf, top_recv_buf, bottom_recv_buf,
                  w, MPI_tag, A1, B1, h1, h2, x_i, y_i);         

        double x_min = A1 + h1 * x_i, y_min = B1 + h2 * y_i;    
        A_mul_vec(x_n, y_n, w, Aw, h1, h2, x_min, y_min);
        
        A_mul_vec_fill_border(x_n, y_n, w, Aw, h1, h2, x_min, y_min, 
                              x_n, y_n, left_flag, right_flag, top_flag, bottom_flag);


        # pragma acc parallel 
        {
        # pragma acc loop independent
        for (int i = 0; i <= x_n + 1; i++) {
            r[i][0] = 0;
            r[i][y_n + 1] = 0;
        }
    
        # pragma acc loop independent
        for (int j = 0; j <= y_n + 1; j++) {
            r[0][j] = 0;
            r[x_n + 1][j] = 0;
        }
        

        # pragma acc loop independent
        for (int i = 1; i <= x_n; i++) {
            # pragma acc loop independent
            for (int j = 1; j <= y_n; j++) {
                r[i][j] = Aw[i][j] - B[i][j];
            }
        }
        }
        
        sync_borders(x_n, y_n, MPI_COMM_CARTESIAN_TOPOLOGY, left_flag, right_flag, 
                  top_flag, bottom_flag, process_dims, curr_coords,
                  left_send_buf, right_send_buf, top_send_buf, bottom_send_buf,
                  left_recv_buf, right_recv_buf, top_recv_buf, bottom_recv_buf,
                  r, MPI_tag, A1, B1, h1, h2, x_i, y_i); 
        
        
        A_mul_vec(x_n, y_n, r, Ar, h1, h2, x_min, y_min);
        A_mul_vec_fill_border(x_n, y_n, r, Ar, h1, h2, x_min, y_min, 
                              x_n, y_n, left_flag, right_flag, top_flag, bottom_flag);
       
        
        tau_norm = norm(x_n, y_n, Ar, h1, h2, left_flag, right_flag, top_flag, bottom_flag);
        tau_dot = dot_product(x_n, y_n, Ar, r, h1, h2, left_flag, right_flag, top_flag, bottom_flag);

        MPI_Allreduce(&tau_norm, &TAU_NORM, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CARTESIAN_TOPOLOGY);
        MPI_Allreduce(&tau_dot, &TAU_DOT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CARTESIAN_TOPOLOGY);
        TAU = TAU_DOT / TAU_NORM;
        
        # pragma acc parallel 
        {
        # pragma acc loop independent
        for (int i = 1; i <= x_n; i++) {
            # pragma acc loop independent
            for (int j = 1; j <= y_n; j++) {
                w_next[i][j] = w[i][j] - TAU * r[i][j];
                w_diff[i][j] = w_next[i][j] - w[i][j];
                w[i][j] = w_next[i][j];
            }
        }
        }

        criteria = sqrt(norm(x_n, y_n, w_diff, h1, h2, left_flag, right_flag, top_flag, bottom_flag));
        MPI_Allreduce(&criteria, &CRITERIA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CARTESIAN_TOPOLOGY);

        converge = CRITERIA < EPSILON;

        n_iter++;
        if (curr_rank == 0) {
            if (n_iter % 1000 == 0) {
                printf("iter %d criteria %3.8f dot %3.8f norm %3.8f tau %3.8f \n", 
                  n_iter, CRITERIA, TAU_DOT, TAU_NORM, TAU);
            }
        }
        
    }
    }

    double fin_timestamp = MPI_Wtime();
    if (curr_rank == 0) {
        printf("Я ВСЕ (rank %d, %d итераций, время %f сек)\n", curr_rank, n_iter, fin_timestamp - start_timestamp);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(left_send_buf); free(left_recv_buf);
    free(right_send_buf); free(right_recv_buf);
    free(top_send_buf); free(top_recv_buf);
    free(bottom_send_buf); free(bottom_recv_buf);
    free(true_u); free(w_diff); free(Ar); free(r); 
    free(Aw); free(w_next); free(w); free(B);

    MPI_Finalize();
    return 0;

}


