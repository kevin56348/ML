#include <iostream>
#include <fstream>
#include "../Eigen/Core"
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

#define FEATURE_NUM 1
#define Y_COL_SIZE 1
#define FUNCTION_LOG "../out.csv"
#define LOG_FILE "../out.log"
#define ERR_LOG "../error.log"

int main(int argc, char **argv)
{
    // redirection
    ofstream clog(LOG_FILE);
    ofstream cerr(ERR_LOG);
    // speed up I/O
    std::ios::sync_with_stdio(false);
    cin.tie(0);
    // function declaration
    pair<double, VectorXd> costFunction(MatrixXd, VectorXd, VectorXd, double);
    VectorXd gradientDescent(MatrixXd, VectorXd, VectorXd, double, size_t, double lambda = 0, double err = 1e-9);
    // reading data and initialize variables
    MatrixXd X;
    VectorXd y, theta, grad;
    size_t m = 0, iter_num = 3000;
    double learning_rate = 0.01, lambda = 0, err = 1e-9;
    theta.conservativeResize(FEATURE_NUM + 1);
    theta = VectorXd::Random(theta.rows());
    {
        ifstream fin;
        string fileName;
        fileName = "../datay.txt";
        fin.open(fileName, std::ios::in);
        if (!fin.is_open())
        {
            cerr << "cannot open file " << fileName << endl;
        }
        double num;
        char buf[1024] = {0};
        while (fin.getline(buf, sizeof(buf)))
        {
            y.conservativeResize(++m, Y_COL_SIZE);
            stringstream word(buf);
            word >> num;
            y(m - 1) = num;
        }
        fin.close();
        // i = row, j = col
        size_t i = 0, j = 0;
        fileName = "../dataX.txt";
        fin.open(fileName, std::ios::in);
        if (!fin.is_open())
        {
            cerr << "cannot open file" << fileName << endl;
        }
        // I don not know what has happened here, the column must greater than
        // Feature_num+1 or it will not work...
        X.conservativeResize(m, FEATURE_NUM + 2);
        while (fin.getline(buf, sizeof(buf)))
        {
            stringstream word(buf);
            // design matrix X
            X(i, j++) = 1;
            while (word >> num)
            {
                X(i, j++) = num;
            }
            j = 0;
            i++;
            if (i > m)
            {
                cerr << "X's column size > y's column size" << endl;
            }
        }
        fin.close();
        X.conservativeResize(m, FEATURE_NUM + 1);
    }
    // gradient descent
    theta = gradientDescent(X, y, theta, learning_rate, iter_num, lambda, err);
    clog << "theta^T: " << theta.transpose() << endl;
    return 0;
}

pair<double, VectorXd> costFunction(MatrixXd X, VectorXd y, VectorXd theta,
                                    double lambda)
{
    // this function will return the cost value and the gradient of theta in
    // a VectorXd, the first element of VectorXd is cost value, the left are
    // the grad of theta.
    VectorXd grad;
    double J = 0;
    size_t m = X.rows();
    grad.conservativeResize(theta.rows());
    J = ((X * theta - y).array().square()).sum() / (2.0 * m) +
        lambda * (theta.array().square().sum() - theta(0) * theta(0)) / (2.0 * m);
    grad = ((X * theta - y).transpose() * X).transpose() / m + (lambda * theta) / m;
    grad(0) -= lambda * theta(0) / m;
    return {J, grad};
}

VectorXd gradientDescent(MatrixXd X, VectorXd y, VectorXd theta,
                         double alpha, size_t iter_num,
                         double lambda = 0, double err = 1e-9)
{
    ofstream clog(FUNCTION_LOG);
    // gradient descent algorithm.
    size_t j = theta.rows();
    VectorXd grad;
    double J;
    pair<double, VectorXd> J_grad;
    // for (size_t k = 0; k < j; k++)
    // {
    //     clog << "theta_" << k << (k == j - 1 ? " " : ",");
    // }
    // clog << endl;
    for (size_t i = 0; i < iter_num; i++)
    {
        J_grad = costFunction(X, y, theta, lambda);
        J = J_grad.first;
        grad = J_grad.second;
        theta -= alpha * grad;

        // for (size_t k = 0; k < j; k++)
        // {
        //     clog << theta(k) << (k == j - 1 ? " " : ",");
        // }
        // clog << endl;
    }
    clog << "succceed\n";

    return theta;
}