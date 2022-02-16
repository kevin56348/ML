#ifndef DATAREAD_H
#define DATAREAD_H

#include "../Eigen/Core"
#include "dataread.cpp"

void file_dimension(std::string file_name, size_t *cols, size_t *rows);
Eigen::MatrixXd read_data(std::string file_name);

#endif