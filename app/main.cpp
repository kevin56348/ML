#include <iostream>
#include <fstream>
#include "../Eigen/Core"
#include "../inc/dataread.h"
#include "../inc/regression.h"
#include "../inc/template.h"
#define VARNAME(value) (#value)
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

#define LOG_FILE "../out.log"
#define ERR_LOG "../error.log"

___MdVdVd_forReturn norm(MatrixXd x)
{
    size_t rows = x.rows(), cols = x.cols();
    ___MdVdVd_forReturn re;
    re.mu.conservativeResize(cols - 1);
    re.sigma.conservativeResize(cols - 1);
    re.norm.conservativeResize(rows, cols);
    for (size_t i = 0; i < cols - 1; i++)
    {
        re.mu(i) = x.col(i + 1).sum() / rows;
        re.sigma(i) = sqrt((x.col(i + 1) - re.mu(i) * VectorXd::Ones(rows)).array().square().sum() / (rows - 1));
        re.norm.col(i + 1) = (x.col(i + 1) - re.mu(i) * VectorXd::Ones(rows)) / re.sigma(i);
    }
    re.norm.col(0) = x.col(0);
    return re;
}
void log(Eigen::VectorXd x, string varname)
{
    ofstream clog(LOG_FILE, std::ios::app);
    for (size_t k = 0; k < x.size(); k++)
    {
        clog << varname << k << (k == x.size() - 1 ? " " : ",");
    }
    clog << endl
         << x(0);
    for (size_t i = 1; i < x.size(); i++)
    {
        clog << "," << x(i);
    }
    clog << endl;
    return;
}
int main(int argc, char **argv)
{
    // redirection
    ofstream clog(LOG_FILE, std::ios::app);
    ofstream cerr(ERR_LOG);
    // speed up I/O
    std::ios::sync_with_stdio(false);
    cin.tie(0);

    // reading data and initialize variables
    MatrixXd X;
    VectorXd y, theta, grad, mu, sigma;
    size_t m, iter_num = 5000, features;
    double learning_rate = 0.01, lambda = 0;

    X = read_data("../data.txt");
    m = X.rows();
    features = X.cols() - 1;

    /*
    initialize theta, y
    let X be design matrix.
    */
    theta.conservativeResize(features + 1);
    theta = VectorXd::Zero(theta.rows());
    y = X.col(features);
    for (size_t i = 0; i < features; i++)
    {
        X.col(features - i) = X.col(features - i - 1);
    }
    X.col(0) = VectorXd::Ones(X.rows());

    {
        ___MdVdVd_forReturn result;
        result = norm(X);
        X = result.norm;
        mu = result.mu;
        sigma = result.sigma;
    }

    // gradient descent
    pair<double, VectorXd> (*costFunction)(Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double);
    costFunction = squareErrorFunction;
    theta = gradientDescent(X, y, theta, learning_rate, iter_num, costFunction, lambda);

    log(theta, VARNAME(theta));

    return 0;
}
