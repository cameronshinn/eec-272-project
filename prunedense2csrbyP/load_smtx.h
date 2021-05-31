#pragma once

#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

typedef unsigned int idx_t;
typedef unsigned int offset_t;

template <typename value_t = float>
struct CSR {
    unsigned int nrows, ncols, nnz;
    std::vector<offset_t> row_ptrs;
    std::vector<idx_t> col_idxs;
    std::vector<value_t> values;
};

template <typename rand_t = float>
rand_t get_random(rand_t begin = 0.0f, rand_t end = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(begin, end);
  return (rand_t)dis(gen);
}

void rm_leading_space(std::string &s) {
    if (s[0] == ' ') {
        s.erase(0, 1);
    }
}

template <typename value_t>
void load_smtx(
    const std::string smtx_path,
    CSR<value_t> &out
) {
    std::ifstream smtx_file(smtx_path);
    offset_t row_ptrs_buf;
    idx_t col_idxs_buf;

    if (smtx_file.is_open()) {
        std::string line;  // Buffer for storing file lines

        for (int line_num = 0; line_num < 3; line_num++) {
            // Skip over comment lines
            do {
                std::getline(smtx_file, line);
            } while (line[0] == '%');

            std::istringstream line_stream(line);

            if (line_num == 0) {  // First Line has dimensions and nnz
                std::string buf;
                std::getline(line_stream, buf, ',');
                rm_leading_space(buf);
                out.nrows = std::stoi(buf);
                std::getline(line_stream, buf, ',');
                rm_leading_space(buf);
                out.ncols = std::stoi(buf);
                std::getline(line_stream, buf, ',');
                rm_leading_space(buf);
                out.nnz = std::stoi(buf);
                out.row_ptrs.clear();
                out.col_idxs.clear();
                out.row_ptrs.reserve(out.nrows + 1);
                out.col_idxs.reserve(out.nnz);
                out.values.reserve(out.nnz);
            } else if (line_num == 1) {  // Second line has row pointers
                while (line_stream >> row_ptrs_buf) {
                    out.row_ptrs.push_back(row_ptrs_buf);
                }
            } else if (line_num == 2) {  // Third line has column indices
                while (line_stream >> col_idxs_buf) {
                    out.col_idxs.push_back(col_idxs_buf);
                    out.values.push_back(get_random<value_t>(1.0f, 10.0f));
                }
            }
        }

        smtx_file.close();
    }

    if (out.row_ptrs.size() != out.nrows + 1 ||
        out.col_idxs.size() != out.nnz ||
        out.values.size() != out.nnz) {
        std::ostringstream ss;
        ss << "Number of non-zeroes in \""  // TODO: Make error messages for each case
           << smtx_path
           << "\" does not match the count in the first line";
        throw (std::invalid_argument(ss.str()));
    }
}
