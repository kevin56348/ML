#include <iostream>
#include <fstream>
#include "../Eigen/Core"
#include "../inc/regression.h"
#include "../inc/template.h"
#include "windows.h"
#define VARNAME(value) (#value)
#define LOG_FILE "../out.csv"
#define ERR_LOG "../error.log"
using namespace Eigen;
using std::cerr;
using std::cin;
using std::clog;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::string;
using std::stringstream;
typedef Matrix<long double, Dynamic, Dynamic> MatrixLdD;
typedef long double ldouble;
void file_dimension(string file_name, size_t *cols, size_t *rows)
{
    ldouble num;
    char buf[2048] = {0};
    ifstream fdata;
    fdata.open(file_name, std::ios::in);
    fdata.getline(buf, sizeof(buf));
    stringstream digits(buf);
    while (digits >> num)
    {
        (*cols)++;
    }
    (*rows)++;
    while (fdata.getline(buf, sizeof(buf)))
    {
        (*rows)++;
    }
    fdata.close();
    return;
}
void read_data(string file_name, MatrixLdD *da)
{
    ifstream fdata;
    ldouble num;
    char buf[2048] = {0};
    size_t cols = 0, rows = 0;
    size_t *pcols = &cols, *prows = &rows;
    file_dimension(file_name, pcols, prows);
    fdata.open(file_name, std::ios::out);
    da->conservativeResize(rows, cols);
    for (size_t row = 0; row < rows; row++)
    {
        fdata.getline(buf, sizeof(buf));
        stringstream dgt(buf);
        for (size_t col = 0; col < cols; col++)
        {
            dgt >> num;
            (*da)(row, col) = num;
        }
    }
    fdata.close();
    return;
}
ldouble squareErrorFunction(MatrixLdD *X, MatrixLdD *y, MatrixLdD *theta, ldouble lambda, MatrixLdD *grad)
{
    // this function will return the cost value and the gradient of theta in
    // a VectorXd, the first element of VectorXd is cost value, the left are
    // the grad of theta.
    ldouble J = 0;
    size_t m = y->rows();
    grad->resize(theta->rows(), theta->cols());
    J = (*X * *theta - *y).array().square().sum() / (2.0 * m) +
        lambda * ((*theta).array().square().sum() - (*theta)(0) * (*theta)(0)) / (2.0 * m);
    *grad = ((*X * *theta - *y).transpose() * *X).transpose() / m + lambda * *theta / m;
    (*grad)(0) -= lambda * (*theta)(0) / m;
    return J;
}
void gradientDescent(MatrixLdD *X, MatrixLdD *y, MatrixLdD *theta,
                     ldouble alpha, size_t iter_num,
                     ldouble (*func)(MatrixLdD *X, MatrixLdD *y, MatrixLdD *theta, ldouble lambda, MatrixLdD *grad),
                     ldouble lambda, MatrixLdD *grad)
{
    ofstream clog(LOG_FILE, std::ios::out);
    size_t j = theta->rows();
    ldouble J;
    for (size_t i = 0; i < iter_num; i++)
    {
        J = func(X, y, theta, lambda, grad); // update grad
        *theta -= alpha * *grad;
        size_t u = 25 * (i + 1) / iter_num;
        cout.precision(3);
        cout << "[" << u * 4.0 << "%]"
             << "  [";
        for (size_t k = 0; k < u; k++)
        {
            cout << "*";
        }
        for (size_t k = 0; k < 25 - u; k++)
        {
            cout << "-";
        }
        cout << "]\r";
        for (size_t k = 0; k < grad->rows(); k++)
        {
            clog << (*grad)(k) << ",";
        }
        clog << J << endl;
    }
    return;
}
void norm(MatrixLdD *x, MatrixLdD *mu, MatrixLdD *sigma)
{
    size_t rows = x->rows(), cols = x->cols();
    MatrixLdD t = MatrixLdD::Ones(rows, cols);
    mu->resize(cols, 1);
    sigma->resize(cols, 1);
    for (size_t i = 1; i < cols; i++)
    {
        (*mu)(i) = (*x).col(i).sum() / rows;
        (*sigma)(i) = sqrt(((*x).col(i) - (*mu)(i)*MatrixLdD::Ones(rows, 1)).array().square().sum() / (rows - 1));
        t.col(i) = ((*x).col(i) - (*mu)(i)*MatrixLdD::Ones(rows, 1)) / (*sigma)(i);
    }
    *x = t;
    return;
}
MatrixLdD sigmoid(MatrixLdD *a)
{
    return (1 / (1 + ((-(*a)).array().exp()).array()).array());
}
double sigmoid(double a)
{
    return (1 / (1 + exp(-a)));
}
ldouble logisticCostFunction(MatrixLdD *X, MatrixLdD *y, MatrixLdD *theta, ldouble lambda, MatrixLdD *grad)
{
    ldouble J = 0;
    size_t m = y->rows();
    MatrixLdD u = *X * *theta;
    u = sigmoid(&u);
    J = -(y->transpose() * (MatrixLdD)(u.array().log()) +
          (MatrixLdD)(1 - y->transpose().array()) * (MatrixLdD)(1 - u.array()).array().log())
             .sum() /
        m;
    ldouble t = (*theta)(0);
    (*theta)(0) = 0;
    J += lambda / (2.0 * m) * (*theta).array().square().sum();
    *grad = 1.0 / m * ((u - *y).transpose() * *X).transpose() + lambda / m * *theta;
    (*theta)(0) = t;
    return J;
}
int main(int argc, char **argv)
{
    ofstream clog(LOG_FILE, std::ios::in);
    ofstream cerr(ERR_LOG);
    std::ios::sync_with_stdio(false);
    cin.tie(0);
    MatrixLdD X, y, theta, grad, mu, sigma;
    size_t m, iter_num = atoi(argv[1]), features;
    ldouble learning_rate = atof(argv[2]), lambda = 0;
    read_data("../data.txt", &X);
    m = X.rows();
    features = X.cols() - 1;
    theta = MatrixLdD::Zero(features + 1, 1);
    grad = MatrixLdD::Zero(theta.rows(), 1);
    y = X.col(features);
    for (size_t i = 0; i < features; i++)
    {
        X.col(features - i) = X.col(features - i - 1);
    }
    for (size_t i = 0; i < m; i++)
    {
        X(i, 0) = (ldouble)1;
    }
    // norm(&X, &mu, &sigma);
    ldouble (*costFunction)(MatrixLdD * X, MatrixLdD * y, MatrixLdD * theta, ldouble lambda, MatrixLdD * grad);
    costFunction = logisticCostFunction;
    unsigned long long start = GetTickCount();
    gradientDescent(&X, &y, &theta, learning_rate, iter_num, costFunction, lambda, &grad);
    // clog << theta.transpose() << endl;
    unsigned long long end = GetTickCount();
    cout << endl
         << end - start;
    return 0;
}