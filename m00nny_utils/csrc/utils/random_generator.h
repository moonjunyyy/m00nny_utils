#ifndef __RANDOM_GENERATOR_H__
#define __RANDOM_GENERATOR_H__

#include <random>
#include <string>
#include <sstream>
#include <array>
#include <algorithm>
#include <type_traits>

namespace utils::random {
static const std::string hex_charset        = "0123456789abcdef";
static const std::string base64_charset     = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const std::string base64_url_charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
static const std::string base32_charset     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
static const std::string base32_hex_charset = "0123456789ABCDEFGHIJKLMNOPQRSTUV";

static const std::wstring hex_wcharset        = L"0123456789abcdef";
static const std::wstring base64_wcharset     = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const std::wstring base64_url_wcharset = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
static const std::wstring base32_wcharset     = L"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
static const std::wstring base32_hex_wcharset = L"0123456789ABCDEFGHIJKLMNOPQRSTUV";

class RandomGenerator {
private:
    std::random_device rd;
    std::mt19937 generator;
    
public: 

    RandomGenerator() : generator(rd()) {}
    explicit RandomGenerator(std::uint32_t seed) : generator(seed) {}
    

    template <typename _F>
    std::enable_if_t<std::is_floating_point_v<_F>>
    uniform(_F min, _F max) { std::uniform_real_distribution<_F> dist(min, max); return dist(generator); }
    template <typename _F>
    std::enable_if_t<std::is_floating_point_v<_F>>
    _normal (_F mean, _F stddev) { std::normal_distribution<_F> dist(mean, stddev); return dist(generator); }
    

    template <typename _I>
    std::enable_if_t<std::is_integral_v<_I>>
    uniform(_I min, _I max) { std::uniform_int_distribution<_I> dist(min, max); return dist(generator); }
    template <typename _I>
    std::enable_if_t<std::is_integral_v<_I>>
    normal (_I mean, _I stddev) { std::normal_distribution<_I> dist(mean, stddev); return dist(generator); }
    template <typename _I>
    std::vector<std::enable_if_t<std::is_integral_v<_I>>> permutation(_I n) {
        std::vector<_I> vec(n);
        std::iota(vec.begin(), vec.end(), 0);
        std::shuffle(vec.begin(), vec.end(), generator);
        return vec;
    }

    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, char> || std::is_same_v<CharT, std::string>, std::string>
    string(size_t length, const std::basic_string<CharT>& charset = "") {
        std::basic_string<CharT> result;
        result.reserve(length);
        if (charset.empty()) for (size_t i = 0; i < length; ++i) { result.push_back((this->uniform<CharT>(0, 255))); }
        else for (size_t i = 0; i < length; ++i) { result.push_back(charset[this->uniform<size_t>(0, charset.size() - 1)]); }
        for (size_t i = 0; i < length; ++i) { result.push_back(charset[dist(generator)]); }
        return result;
    }

    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, wchar_t> || std::is_same_v<CharT, std::wstring>, std::wstring>
    string(size_t length, const std::basic_string<CharT>& charset = "") {
        std::basic_string<CharT> result;
        result.reserve(length);
        if (charset.empty()) for (size_t i = 0; i < length; ++i) { result.push_back((this->uniform<CharT>(0, 255))); }
        else for (size_t i = 0; i < length; ++i) { result.push_back(charset[this->uniform<size_t>(0, charset.size() - 1)]); }
        for (size_t i = 0; i < length; ++i) { result.push_back(charset[dist(generator)]); }
        return result;
    }

    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, char> || std::is_same_v<CharT, std::string>, std::string>
    hex(size_t length) { return string(length, hex_charset); }
    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, char> || std::is_same_v<CharT, std::string>, std::string>
    base64(size_t length) { return string(length, base64_charset); }
    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, char> || std::is_same_v<CharT, std::string>, std::string>
    base64url(size_t length) { return string(length, base64_url_charset); }
    template <typename CharT>
    std::enable_if_t<std::is_same_v<CharT, char> || std::is_same_v<CharT, std::string>, std::string>
    uuid() {
        std::array<uint8_t, 16> data;
        std::generate(data.begin(), data.end(), [this]() { this->uniform<uint8_t>(0, 255); });
        data[6] = (data[6] & 0x0F) | 0x40; // version 4
        data[8] = (data[8] & 0x3F) | 0x80; // transform to RFC 4122 variant
        
        std::basic_stringstream<CharT> ss;
        ss << std::hex << std::setfill('0');
        
        for (size_t i = 0; i < 16; ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10) ss << '-';
            ss << std::setw(2) << static_cast<int>(data[i]);
        }
        return ss.str();
    }


    template <typename CharT>
    std::basic_string<std::enable_if_t<std::is_same_v<CharT, wchar_t>>, std::wstring>
    hex(size_t length) { return string(length, hex_wcharset); }
    template <typename CharT>
    std::basic_string<std::enable_if_t<std::is_same_v<CharT, wchar_t>>, std::wstring>
    base64(size_t length) { return string(length, base64_wcharset); }
    template <typename CharT>
    std::basic_string<std::enable_if_t<std::is_same_v<CharT, wchar_t>>, std::wstring>
    base64url(size_t length) { return string(length, base64_url_wcharset); }
    template <typename CharT>
    std::basic_string<std::enable_if_t<std::is_same_v<CharT, wchar_t>>, std::wstring>
    uuid() {
        std::array<uint8_t, 16> data;
        std::generate(data.begin(), data.end(), [this]() { this->uniform<uint8_t>(0, 255); });
        data[6] = (data[6] & 0x0F) | 0x40; // version 4
        data[8] = (data[8] & 0x3F) | 0x80; // transform to RFC 4122 variant
        
        std::basic_stringstream<CharT> ss;
        ss << std::hex << std::setfill('0');
        
        for (size_t i = 0; i < 16; ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10) ss << '-';
            ss << std::setw(2) << static_cast<int>(data[i]);
        }
        return ss.str();
    }

    template <typename Container>
    auto& choose(Container& container) {
        if (container.empty()) throw std::logic_error("Cannot choose from empty container");
        std::uniform_int_distribution<size_t> dist(0, container.size() - 1);
        auto it = container.begin();
        std::advance(it, dist(generator));
        return *it;
    }
    template <typename Container>
    void shuffle(Container& container) { std::shuffle(container.begin(), container.end(), generator); }
    std::mt19937& getGenerator() { return generator; }
};
// Singleton instance for global RandomGenerator
RandomGenerator& getGlobalRandomGenerator() { static RandomGenerator instance; return instance; }
};  // namespace utils::random
#endif // __RANDOM_GENERATOR_H__