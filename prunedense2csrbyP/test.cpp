#include <iostream>

#include "load_smtx.hpp"

int main (int argc, char *argv[]) {
    std::vector<std::string> smtx_paths(argv + 1, argv + argc);

    for (auto &p : smtx_paths) {
        CSR<float> spm;
        load_smtx(p, spm);
        std::cout << spm.nnz << std::endl;
    }
}
