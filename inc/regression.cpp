#include <fstream>
#include <iostream>
#include <vector>
#include "../Eigen/Core"
#include "dataread.h"

std::pair<double, Eigen::VectorXd> squareErrorFunction(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd theta, double lambda)
{
    // this function will return the cost value and the gradient of theta in
    // a VectorXd, the first element of VectorXd is cost value, the left are
    // the grad of theta.
    Eigen::VectorXd grad;
    double J = 0;
    size_t m = X.rows();
    grad.conservativeResize(theta.rows());
    J = ((X * theta - y).array().square()).sum() / (2.0 * m) +
        lambda * (theta.array().square().sum() - theta(0) * theta(0)) / (2.0 * m);
    grad = ((X * theta - y).transpose() * X).transpose() / m + (lambda * theta) / m;
    grad(0) -= lambda * theta(0) / m;
    return {J, grad};
}
Eigen::VectorXd gradientDescent(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd theta,
                                double alpha, size_t iter_num,
                                std::pair<double, Eigen::VectorXd> (*func)(Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double),
                                double lambda)
{
    // gradient descent algorithm.
    size_t j = theta.rows();
    Eigen::VectorXd grad, preTheta;
    double J;
    std::pair<double, Eigen::VectorXd> J_grad;
    for (size_t i = 0; i < iter_num; i++)
    {
        J_grad = func(X, y, theta, lambda);
        J = J_grad.first;
        grad = J_grad.second;
        theta -= alpha * grad;
    }
    return theta;
}