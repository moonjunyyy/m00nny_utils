#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <optional>
#include <functional>
#include <filesystem>
#include <fstream>

namespace utils {
// 바이트 수를 사람이 읽기 쉬운 형태로 변환하는 유틸리티 함수
std::string bytesToHumanReadable(std::uintmax_t size) {
    constexpr std::uintmax_t KB = 1024;
    constexpr std::uintmax_t MB = 1024 * KB;
    constexpr std::uintmax_t GB = 1024 * MB;
    constexpr std::uintmax_t TB = 1024 * GB;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    if (size < KB) {
        ss << size << " B";
    } else if (size < MB) {
        ss << static_cast<double>(size) / KB << " KB";
    } else if (size < GB) {
        ss << static_cast<double>(size) / MB << " MB";
    } else if (size < TB) {
        ss << static_cast<double>(size) / GB << " GB";
    } else {
        ss << static_cast<double>(size) / TB << " TB";
    }
    return ss.str();
}
}; // namespace utils
#endif // __UTILS_H__