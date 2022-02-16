#include <fstream>
#include <iostream>
#include <vector>
#include "../Eigen/Core"
#include "dataread.h"

void file_dimension(std::string file_name, size_t *cols, size_t *rows)
{
    double num;
    char buf[2048] = {0};
    std::ifstream fdata;
    fdata.open(file_name, std::ios::in);
    fdata.getline(buf, sizeof(buf));
    std::stringstream digits(buf);
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
Eigen::MatrixXd read_data(std::string file_name)
{
    Eigen::MatrixXd data;
    std::ifstream fdata;
    double num;
    char buf[2048] = {0};
    size_t cols = 0, rows = 0;
    size_t *pcols = &cols, *prows = &rows;
    file_dimension(file_name, pcols, prows);
    fdata.open(file_name, std::ios::in);
    data.conservativeResize(rows, cols);
    for (size_t row = 0; row < rows; row++)
    {
        fdata.getline(buf, sizeof(buf));
        std::stringstream dgt(buf);
        for (size_t col = 0; col < cols; col++)
        {
            dgt >> num;
            data(row, col) = num;
        }
    }
    fdata.close();
    return data;
}
