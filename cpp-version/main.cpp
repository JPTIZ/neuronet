#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "utils.h"

using namespace std::string_literals;
using namespace utils;

using Input = std::vector<int>;
using Output = std::vector<int>;

auto from_csv(const std::string& filename) {
    auto file = std::ifstream{filename};
    auto [inputs, outputs] = std::pair<std::vector<Input>, std::vector<Output>>{};

    auto line = ""s;
    while (std::getline(file, line)) {
        auto elms = split(line, ",");
        inputs.push_back(slice<std::string, int>(elms, 0, elms.size(), stoi));

        outputs.push_back({std::stoi(elms[0])});
    }

    return std::pair{inputs, outputs};
}

int main(int argc, char* argv[]) {
    auto [inputs, outputs] = from_csv(argv[1]);
}
