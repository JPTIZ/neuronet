#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <functional>
#include <string>
#include <vector>

namespace utils {

std::vector<std::string> split(const std::string&, const std::string&);

/*
template <typename T>
std::vector<T> slice(const std::vector<T>&, int, int);
*/

template <typename T, typename U>
std::vector<U> slice(const std::vector<T>&, int, int, std::function<U(const T&)> = T());

template <typename T, typename U>
std::vector<U> slice(const std::vector<T>& v, std::function<U(const T&)> = T());

inline auto stoi(const std::string& s) {
    return std::stoi(s);
}

}

#endif
