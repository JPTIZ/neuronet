#include "utils.h"

#include <regex>

std::vector<std::string> utils::split(const std::string& str, const std::string& delim) {
    std::regex regex{delim};
    std::sregex_token_iterator
        first(str.begin(), str.end(), regex, -1),
        last{};
    return {first, last};
}


template <typename T, typename U>
std::vector<U> utils::slice(const std::vector<T>& v, int start, int end, std::function<U(T)> f) {
    auto result = std::vector<U>{};

    for (auto i = start; i < end; ++i) {
        result.push_back(f(v[i]));
    }

    return result;
}


template <typename T, typename U>
std::vector<U> utils::slice(const std::vector<T>& v, std::function<U(T)>) {
    return slice(v, 0, v.size());
}
