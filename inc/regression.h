#ifndef REGRESSION_H
#define REGRESSION_H
#include "../Eigen/Core"
#include "regression.cpp"

std::pair<double, Eigen::VectorXd> squareErrorFunction(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd theta, double lambda);
Eigen::VectorXd gradientDescent(Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double, size_t,
                                std::pair<double, Eigen::VectorXd> (*func)(Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double),
                                double);

#endif