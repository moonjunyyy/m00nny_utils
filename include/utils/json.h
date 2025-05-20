#ifndef __JSON_H__
#define __JSON_H__

#include "utils.h"
#include <map>
#include <variant>
#include <variant>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <iostream> // For error reporting / debugging
#include <sstream>  // For number to string, string to number
#include <fstream>  // For file operations
#include <algorithm> // For std::remove_if
#include <iomanip>  // For std::setprecision with doubles
#include <cctype>   // For isspace, isdigit etc.

// Your utils::task might be used for async file load/save if desired, but not core to JSON logic itself.

namespace json
{
// Forward declaration
class JsonData;

// Type aliases
using JsonArray  = std::vector<JsonData>;
using JsonObject = std::map<std::string, JsonData>; // std::map for ordered keys (JSON spec doesn't require order, but it's common)
                                                // For unordered, std::unordered_map could be an option.
// IMPROVEMENT: Added JsonNumber as a more flexible number type internally for parsing,
// then convert to long long or double. Or, just parse directly. For simplicity, let's parse directly.
using JsonValue = std::variant<
    std::nullptr_t, // null
    bool,           // true, false
    long long,      // integers
    double,         // floating-point numbers
    std::string,    // strings
    JsonArray,      // arrays
    JsonObject      // objects
>;

// Enum for type identification
enum class JsonDataType
{
    JSON_NULL,
    JSON_BOOLEAN,
    JSON_INTEGER,
    JSON_FLOAT,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
    JSON_INVALID // Should ideally not be used if parsing is correct
};

// Custom exception for JSON specific errors
class JsonException : public std::runtime_error {
public:
    JsonException(const std::string& msg) : std::runtime_error(msg) {}
};


class JsonData
{
private:
    JsonValue value_; // IMPROVEMENT: Suffix underscore for private members

public:
    // --- Constructors ---
    JsonData() : value_(nullptr) {} // Default to null

    // Generic constructor for underlying types
    // IMPROVEMENT: SFINAE to ensure T is one of the variant's alternatives or convertible.
    // This is tricky with char* -> string conversion. Your current one is mostly fine.
    template <typename T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, JsonData> &&
                                          !std::is_constructible_v<JsonData, T&&, bool>>> // Avoid ambiguity with other constructors
    JsonData(T&& val) : value_(std::forward<T>(val)) {}

    // Specific constructor for const char* to avoid ambiguity and ensure it becomes std::string
    JsonData(const char* c_str) : value_(std::string(c_str)) {}

    // Copy and Move semantics (already good)
    JsonData(const JsonData& other)     = default;
    JsonData(JsonData&& other) noexcept = default;
    JsonData& operator=(const JsonData& other)     = default;
    JsonData& operator=(JsonData&& other) noexcept = default;
    ~JsonData() = default;

    // --- Type Information ---
    JsonDataType getType() const {
        // Your existing getType is good.
        return std::visit([](auto&& arg) -> JsonDataType {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) return JsonDataType::JSON_NULL;
            else if constexpr (std::is_same_v<T, bool>) return JsonDataType::JSON_BOOLEAN;
            else if constexpr (std::is_same_v<T, long long>) return JsonDataType::JSON_INTEGER;
            else if constexpr (std::is_same_v<T, double>) return JsonDataType::JSON_FLOAT;
            else if constexpr (std::is_same_v<T, std::string>) return JsonDataType::JSON_STRING;
            else if constexpr (std::is_same_v<T, JsonArray>) return JsonDataType::JSON_ARRAY;
            else if constexpr (std::is_same_v<T, JsonObject>) return JsonDataType::JSON_OBJECT;
            // This path should ideally not be reached if JsonValue only contains valid types.
            return JsonDataType::JSON_INVALID;
        }, value_);
    }

    template <typename T>
    bool is() const { return std::holds_alternative<T>(value_); }

    // --- Value Access ---
    // IMPROVEMENT: More specific error messages and use JsonException
    template <typename T>
    const T& get() const {
        try {
            return std::get<T>(value_);
        } catch (const std::bad_variant_access&) {
            throw JsonException("Invalid type access in JsonData::get<const T&>(). Requested type does not match stored type. Stored type: " + typeToString(getType()));
        }
    }

    template <typename T>
    T& get() {
        try {
            return std::get<T>(value_);
        } catch (const std::bad_variant_access&) {
            throw JsonException("Invalid type access in JsonData::get<T&>(). Requested type does not match stored type. Stored type: " + typeToString(getType()));
        }
    }
    
    // IMPROVEMENT: Optional-like access (returns pointer or nullptr)
    template <typename T>
    const T* get_if() const noexcept { return std::get_if<T>(&value_); }

    template <typename T>
    T* get_if() noexcept { return std::get_if<T>(&value_); }


    // --- Value Modification ---
    template <typename T>
    void set(T&& new_value) { value_ = std::forward<T>(new_value); }
    void set(const char* c_str) { value_ = std::string(c_str); } // Overload for const char*

    // --- Array Access ---
    // IMPROVEMENT: More robust array access, ensuring type.
    JsonData& operator[](size_t index) {
        if (is<std::nullptr_t>() || getType() == JsonDataType::JSON_INVALID) { // If null or uninitialized, make it an array
            value_ = JsonArray();
        } else if (!is<JsonArray>()) {
            throw JsonException("Cannot use operator[] (index) on non-array type. Current type: " + typeToString(getType()));
        }
        JsonArray& arr = std::get<JsonArray>(value_);
        if (index >= arr.size()) {
            arr.resize(index + 1); // Default construct new JsonData (which are null)
        }
        return arr[index];
    }

    // Const array access
    const JsonData& operator[](size_t index) const {
        if (!is<JsonArray>()) {
            throw JsonException("Cannot use operator[] (index) on non-array type. Current type: " + typeToString(getType()));
        }
        const JsonArray& arr = std::get<JsonArray>(value_);
        if (index >= arr.size()) {
            throw std::out_of_range("Index out of range in const JsonData::operator[]");
        }
        return arr[index];
    }
    
    // IMPROVEMENT: Explicit array element access with bounds checking for const version
    const JsonData& at(size_t index) const {
        return (*this)[index]; // Relies on the bounds check in const operator[]
    }

    // --- Object Access ---
    // IMPROVEMENT: More robust object access
    JsonData& operator[](const std::string& key) {
         if (is<std::nullptr_t>() || getType() == JsonDataType::JSON_INVALID) { // If null or uninitialized, make it an object
            value_ = JsonObject();
        } else if (!is<JsonObject>()) {
            throw JsonException("Cannot use operator[] (key) on non-object type. Current type: " + typeToString(getType()));
        }
        return std::get<JsonObject>(value_)[key]; // Default constructs if key doesn't exist
    }
    
    JsonData& operator[](const char* key) { // Convenience for char* keys
        return (*this)[std::string(key)];
    }

    // Const object access (no creation)
    // IMPROVEMENT: Renamed your at() to operator[] for consistency, created a new at()
    const JsonData& operator[](const std::string& key) const {
        if (!is<JsonObject>()) {
            throw JsonException("Cannot use operator[] (key) on non-object type. Current type: " + typeToString(getType()));
        }
        const JsonObject& obj = std::get<JsonObject>(value_);
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::out_of_range("Key not found in const JsonData::operator[]: " + key);
        }
        return it->second;
    }
     const JsonData& operator[](const char* key) const { // Convenience for char* keys
        return (*this)[std::string(key)];
    }


    // IMPROVEMENT: `at` for objects now consistently means checked access for const.
    const JsonData& at(const std::string& key) const {
        return (*this)[key]; // Relies on the bounds check in const operator[]
    }
    const JsonData& at(const char* key) const {
        return (*this)[std::string(key)];
    }

    // --- Utility / Other ---
    bool hasKey(const std::string& key) const {
        if (!is<JsonObject>()) return false;
        const auto& obj = std::get<JsonObject>(value_);
        return obj.count(key) > 0;
    }

    size_t size() const { // Returns array size or object member count
        if (is<JsonArray>()) return std::get<JsonArray>(value_).size();
        if (is<JsonObject>()) return std::get<JsonObject>(value_).size();
        return 0; // Or throw for other types
    }

    bool empty() const { return size() == 0; }

    void clear() { // Clears array or object, or sets to null
        if (is<JsonArray>()) std::get<JsonArray>(value_).clear();
        else if (is<JsonObject>()) std::get<JsonObject>(value_).clear();
        else value_ = nullptr;
    }


    // --- Serialization (toString) ---
    // Your escapeString is good.
    static std::string escapeString(const std::string& input) { // IMPROVEMENT: Made static as it doesn't depend on member data
        std::string output;
        output.reserve(input.length());
        for (char c : input) {
            switch (c) {
                case '"':  output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b";  break;
                case '\f': output += "\\f";  break;
                case '\n': output += "\\n";  break;
                case '\r': output += "\\r";  break;
                case '\t': output += "\\t";  break;
                default:
                    // Control characters (U+0000 to U+001F) must be escaped
                    if (static_cast<unsigned char>(c) <= 0x1F) {
                        char hex_chars[7]; // \uXXXX + null terminator
                        snprintf(hex_chars, sizeof(hex_chars), "\\u%04x", static_cast<unsigned char>(c));
                        output += hex_chars;
                    } else {
                        output += c;
                    }
                    break;
            }
        }
        return output;
    }

    // Your toString is mostly good. Minor tweaks for precision and clarity.
    std::string toString(bool pretty_print = false, const std::string& indent_unit = "  ", int current_level = 0) const {
        std::string current_indent_str;
        if (pretty_print) {
            for (int i = 0; i < current_level; ++i) {
                current_indent_str += indent_unit;
            }
        }

        const std::string nl = pretty_print ? "\n" : "";
        const std::string sp_after_colon = pretty_print ? " " : "";
        const std::string sp_element = pretty_print ? " " : ""; // Optional space after comma for elements

        return std::visit(
            [&](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                std::string result_str;

                if constexpr (std::is_same_v<T, std::nullptr_t>) {
                    result_str = "null";
                } else if constexpr (std::is_same_v<T, bool>) {
                    result_str = arg ? "true" : "false";
                } else if constexpr (std::is_same_v<T, long long>) {
                    result_str = std::to_string(arg);
                } else if constexpr (std::is_same_v<T, double>) {
                    std::ostringstream oss;
                    // IMPROVEMENT: Control precision for doubles to avoid excessive digits
                    // and ensure a decimal point for whole numbers to distinguish from integers if desired.
                    oss << std::fixed << std::setprecision(15) << arg; // Adjust precision as needed
                    result_str = oss.str();
                    // Remove trailing zeros and a trailing decimal point if it's like "1.0" -> "1" (optional)
                    result_str.erase(result_str.find_last_not_of('0') + 1, std::string::npos);
                    if (result_str.back() == '.') {
                        result_str.pop_back();
                    }
                } else if constexpr (std::is_same_v<T, std::string>) {
                    result_str = "\"" + JsonData::escapeString(arg) + "\"";
                } else if constexpr (std::is_same_v<T, JsonArray>) {
                    result_str += "[";
                    if (!arg.empty()) {
                        if (pretty_print) result_str += nl;
                        bool first_element = true;
                        for (const auto& item : arg) {
                            if (!first_element) {
                                result_str += ",";
                                if (pretty_print) result_str += sp_element; // Space after comma
                            }
                            if (pretty_print && !first_element) result_str += nl;

                            if (pretty_print) result_str += current_indent_str + indent_unit;
                            result_str += item.toString(pretty_print, indent_unit, current_level + 1);
                            first_element = false;
                        }
                        if (pretty_print) result_str += nl;
                    }
                    if (pretty_print && !arg.empty()) result_str += current_indent_str;
                    result_str += "]";
                } else if constexpr (std::is_same_v<T, JsonObject>) {
                    result_str += "{";
                    if (!arg.empty()) {
                        if (pretty_print) result_str += nl;
                        bool first_member = true;
                        for (const auto& pair : arg) {
                            if (!first_member) {
                                result_str += ",";
                                if (pretty_print) result_str += sp_element; // Space after comma
                            }
                             if (pretty_print && !first_member) result_str += nl;

                            if (pretty_print) result_str += current_indent_str + indent_unit;
                            result_str += "\"" + JsonData::escapeString(pair.first) + "\":" + sp_after_colon;
                            result_str += pair.second.toString(pretty_print, indent_unit, current_level + 1);
                            first_member = false;
                        }
                        if (pretty_print) result_str += nl;
                    }
                    if (pretty_print && !arg.empty()) result_str += current_indent_str;
                    result_str += "}";
                }
                return result_str;
            },
            value_
        );
    }

// --- Helper for pretty printing type ---
    static std::string typeToString(JsonDataType type) {
        switch (type) {
            case JsonDataType::JSON_NULL:    return "null";
            case JsonDataType::JSON_BOOLEAN: return "boolean";
            case JsonDataType::JSON_INTEGER: return "integer";
            case JsonDataType::JSON_FLOAT:   return "float";
            case JsonDataType::JSON_STRING:  return "string";
            case JsonDataType::JSON_ARRAY:   return "array";
            case JsonDataType::JSON_OBJECT:  return "object";
            default:                         return "invalid";
        }
    }
};


// Your getJsonTypeEnum seems like a helper that might not be directly needed
// if JsonData::getType() is used. It could be a free function if desired.


// --- Parser Helper Structure ---
struct ParserState {
    const char* current;
    const char* end;
    int line_number = 1;
    int column_number = 1;

    ParserState(const std::string& str) : current(str.data()), end(str.data() + str.size()) {}
    ParserState(const char* s, const char* e) : current(s), end(e) {}


    char peek() const {
        if (current >= end) return '\0';
        return *current;
    }

    char advance() {
        if (current >= end) return '\0';
        char c = *current++;
        if (c == '\n') {
            line_number++;
            column_number = 1;
        } else {
            column_number++;
        }
        return c;
    }

    bool match(char expected) {
        if (peek() == expected) {
            advance();
            return true;
        }
        return false;
    }

    void skipWhitespaceAndComments() {
        while (true) {
            char c = peek();
            if (isspace(c)) {
                advance();
            } else if (c == '/' && (current + 1 < end && *(current + 1) == '/')) { // Line comments
                advance(); advance(); // Skip //
                while (peek() != '\n' && peek() != '\0') advance();
            } else if (c == '/' && (current + 1 < end && *(current + 1) == '*')) { // Block comments
                advance(); advance(); // Skip /*
                while (true) {
                    if (peek() == '\0') throw JsonException("Unterminated block comment at L" + std::to_string(line_number) + " C" + std::to_string(column_number));
                    if (advance() == '*' && peek() == '/') {
                        advance(); // Skip /
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
    }

    JsonException error(const std::string& message) const {
        return JsonException(message + " (at L" + std::to_string(line_number) + " C" + std::to_string(column_number) + ")");
    }
};

// Forward declarations for recursive parsing functions
JsonData parseValue(ParserState& s);
JsonArray parseArray(ParserState& s);
JsonObject parseObject(ParserState& s);
std::string parseString(ParserState& s);
JsonData parseNumber(ParserState& s);
JsonData parseLiteral(ParserState& s, const std::string& literal, JsonData value_on_match);


// --- Main JSON Class (Parser and File I/O) ---
class Json : public JsonData // Json IS A JsonData
{
public:
    Json() = default; // Inherits JsonData's default (nullptr)

    // Parse from string
    explicit Json(const std::string& json_string) {
        if (!tryParse(json_string, *this)) {
             // Error already thrown by tryParse if it fails critically
        }
    }
    explicit Json(const char* json_c_string) {
        if (!tryParse(std::string(json_c_string), *this)) {
            // Error
        }
    }

    // Parse from input stream
    explicit Json(std::istream& input_stream) {
        std::string content((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
        if (input_stream.bad()) throw JsonException("Error reading from input stream.");
        if (!tryParse(content, *this)) {
            // Error
        }
    }
    
    // Construct from existing JsonData types directly
    Json(const JsonData& data) : JsonData(data) {}
    Json(JsonData&& data) : JsonData(std::move(data)) {}
    Json(const JsonObject& obj) : JsonData(obj) {}
    Json(JsonObject&& obj) : JsonData(std::move(obj)) {}
    Json(const JsonArray& arr) : JsonData(arr) {}
    Json(JsonArray&& arr) : JsonData(std::move(arr)) {}
    template <typename T,
              typename = std::enable_if_t<std::is_constructible_v<JsonData, T&&> &&
                                          !std::is_same_v<std::decay_t<T>, Json> && // Prevent recursion with self
                                          !std::is_same_v<std::decay_t<T>, std::string> && // Handled by string constructor
                                          !std::is_same_v<std::decay_t<T>, const char*>>> // Handled by c_string constructor
    Json(T&& val) : JsonData(std::forward<T>(val)) {}


    // --- Static parsing methods ---
    static Json fromString(const std::string& json_string) {
        Json json_doc;
        if (!tryParse(json_string, json_doc)) {
             // tryParse throws on failure.
        }
        return json_doc;
    }
    static Json fromStream(std::istream& input_stream) {
        Json json_doc;
        std::string content((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
         if (input_stream.bad()) throw JsonException("Error reading from input stream for fromStream.");
        if (!tryParse(content, json_doc)) {
            // tryParse throws on failure.
        }
        return json_doc;
    }
    static Json fromFile(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw JsonException("Failed to open file for parsing: " + file_path);
        }
        return fromStream(file);
    }

    // --- Instance parsing methods (modify current Json object) ---
    bool parse(const std::string& json_string) {
        return tryParse(json_string, *this);
    }
    bool parse(std::istream& input_stream) {
        std::string content((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
        if (input_stream.bad()) {
            this->set(nullptr); // Clear on read error
            throw JsonException("Error reading from input stream for instance parse.");
        }
        return tryParse(content, *this);
    }
    bool load(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            this->set(nullptr); // Clear on file open error
            throw JsonException("Failed to open file for loading: " + file_path);
        }
        return parse(file);
    }


    // --- Serialization ---
    // Dump to string (inherits toString from JsonData)
    // `dump()` is a common name for this
    std::string dump(bool pretty_print = false, const std::string& indent_unit = "  ") const {
        return this->toString(pretty_print, indent_unit, 0);
    }

    // Save to file
    bool save(const std::string& file_path, bool pretty_print = true, bool overwrite = true) const {
        if (!overwrite && std::filesystem::exists(file_path)) { // Requires <filesystem>
             // std::cerr << "File already exists and overwrite is false: " << file_path << std::endl;
            return false; // Or throw
        }
        std::ofstream file(file_path);
        if (!file.is_open()) {
            throw JsonException("Failed to open file for saving: " + file_path);
        }
        file << dump(pretty_print);
        return file.good();
    }


private:
    // --- Static parsing implementation ---
    // target is the JsonData object to populate
    static bool tryParse(const std::string& json_string, JsonData& target) {
        ParserState s(json_string);
        try {
            s.skipWhitespaceAndComments();
            target = parseValue(s);
            s.skipWhitespaceAndComments();
            if (s.peek() != '\0') { // Check for trailing characters
                throw s.error("Unexpected trailing characters after JSON value");
            }
            return true;
        } catch (const JsonException& e) {
            // std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
            target.set(nullptr); // Ensure target is null on error
            throw; // Re-throw to allow caller to catch
        }
        // Catch other potential exceptions if necessary, though JsonException should cover parsing logic
        return false; // Should not be reached if exceptions are thrown
    }
};


// --- Recursive Descent Parser Implementation ---

JsonData parseValue(ParserState& s) {
    s.skipWhitespaceAndComments();
    char c = s.peek();
    switch (c) {
        case '{': return parseObject(s);
        case '[': return parseArray(s);
        case '"': return parseString(s);
        case 't': return parseLiteral(s, "true", true);
        case 'f': return parseLiteral(s, "false", false);
        case 'n': return parseLiteral(s, "null", nullptr);
        case '-':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return parseNumber(s);
        default:
            throw s.error(std::string("Unexpected character '") + c + "' when expecting a value");
    }
}

JsonData parseLiteral(ParserState& s, const std::string& literal_str, JsonData value_on_match) {
    for (char expected_char : literal_str) {
        if (s.advance() != expected_char) {
            throw s.error("Invalid literal. Expected '" + literal_str + "'");
        }
    }
    return value_on_match;
}


std::string parseString(ParserState& s) {
    if (!s.match('"')) throw s.error("Expected '\"' to start a string");
    std::string str_val;
    while (s.peek() != '"') {
        char c = s.advance();
        if (c == '\0') throw s.error("Unterminated string");
        if (c == '\\') { // Escape sequence
            char escaped = s.advance();
            switch (escaped) {
                case '"':  str_val += '"'; break;
                case '\\': str_val += '\\'; break;
                case '/':  str_val += '/'; break; // JSON allows / to be escaped, though not required
                case 'b':  str_val += '\b'; break;
                case 'f':  str_val += '\f'; break;
                case 'n':  str_val += '\n'; break;
                case 'r':  str_val += '\r'; break;
                case 't':  str_val += '\t'; break;
                case 'u': { // Unicode escape \uXXXX
                    unsigned int codepoint = 0;
                    for (int i = 0; i < 4; ++i) {
                        char hex_char = s.advance();
                        if (hex_char == '\0') throw s.error("Unterminated string in unicode escape");
                        codepoint <<= 4;
                        if (hex_char >= '0' && hex_char <= '9') codepoint += (hex_char - '0');
                        else if (hex_char >= 'a' && hex_char <= 'f') codepoint += (10 + hex_char - 'a');
                        else if (hex_char >= 'A' && hex_char <= 'F') codepoint += (10 + hex_char - 'A');
                        else throw s.error("Invalid hex digit in unicode escape");
                    }
                    // Convert UTF-32 codepoint to UTF-8
                    if (codepoint <= 0x7F) { // 1-byte
                        str_val += static_cast<char>(codepoint);
                    } else if (codepoint <= 0x7FF) { // 2-bytes
                        str_val += static_cast<char>(0xC0 | (codepoint >> 6));
                        str_val += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else if (codepoint <= 0xFFFF) { // 3-bytes
                        // Check for surrogate pairs (U+D800 to U+DFFF), which are invalid alone
                        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
                            throw s.error("Invalid unicode: Lone surrogate U+" + std::to_string(codepoint));
                        }
                        str_val += static_cast<char>(0xE0 | (codepoint >> 12));
                        str_val += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        str_val += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else if (codepoint <= 0x10FFFF) { // 4-bytes
                        str_val += static_cast<char>(0xF0 | (codepoint >> 18));
                        str_val += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
                        str_val += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        str_val += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else {
                        throw s.error("Invalid unicode codepoint U+" + std::to_string(codepoint));
                    }
                    break;
                }
                default: throw s.error(std::string("Invalid escape sequence '\\") + escaped + "'");
            }
        } else {
            str_val += c;
        }
    }
    if (!s.match('"')) throw s.error("Expected '\"' to end a string"); // Should be guaranteed by while loop condition
    return str_val;
}

JsonData parseNumber(ParserState& s) {
    std::string num_str;
    bool is_float = false;

    if (s.peek() == '-') num_str += s.advance();

    // Integer part
    if (s.peek() == '0') {
        num_str += s.advance();
        if (isdigit(s.peek())) throw s.error("Numbers cannot have leading zeros (unless it's just '0')");
    } else {
        while (isdigit(s.peek())) num_str += s.advance();
    }

    // Fractional part
    if (s.peek() == '.') {
        is_float = true;
        num_str += s.advance();
        if (!isdigit(s.peek())) throw s.error("Expected digit after decimal point");
        while (isdigit(s.peek())) num_str += s.advance();
    }

    // Exponent part
    if (s.peek() == 'e' || s.peek() == 'E') {
        is_float = true;
        num_str += s.advance();
        if (s.peek() == '+' || s.peek() == '-') num_str += s.advance();
        if (!isdigit(s.peek())) throw s.error("Expected digit in exponent");
        while (isdigit(s.peek())) num_str += s.advance();
    }

    if (num_str.empty() || (num_str.length() == 1 && num_str[0] == '-')) {
        throw s.error("Invalid number format");
    }
    
    try {
        if (is_float) {
            return std::stod(num_str);
        } else {
            return std::stoll(num_str);
        }
    } catch (const std::out_of_range& oor) {
        throw s.error("Number out of range: " + num_str);
    } catch (const std::invalid_argument& ia) {
        throw s.error("Invalid number format (stod/stoll): " + num_str);
    }
}


JsonArray parseArray(ParserState& s) {
    if (!s.match('[')) throw s.error("Expected '[' to start an array");
    JsonArray arr;
    s.skipWhitespaceAndComments();
    if (s.peek() == ']') { // Empty array
        s.advance();
        return arr;
    }
    while (true) {
        arr.push_back(parseValue(s));
        s.skipWhitespaceAndComments();
        if (s.peek() == ']') {
            s.advance();
            break;
        }
        if (!s.match(',')) throw s.error("Expected ',' or ']' in array");
        s.skipWhitespaceAndComments(); // Allow whitespace after comma before next value
         if (s.peek() == ']') throw s.error("Trailing comma in array is not allowed by strict JSON"); // Strict JSON
    }
    return arr;
}

JsonObject parseObject(ParserState& s) {
    if (!s.match('{')) throw s.error("Expected '{' to start an object");
    JsonObject obj;
    s.skipWhitespaceAndComments();
    if (s.peek() == '}') { // Empty object
        s.advance();
        return obj;
    }
    while (true) {
        s.skipWhitespaceAndComments();
        std::string key = parseString(s);
        s.skipWhitespaceAndComments();
        if (!s.match(':')) throw s.error("Expected ':' after key in object");
        s.skipWhitespaceAndComments();
        obj[key] = parseValue(s);
        s.skipWhitespaceAndComments();
        if (s.peek() == '}') {
            s.advance();
            break;
        }
        if (!s.match(',')) throw s.error("Expected ',' or '}' in object");
        s.skipWhitespaceAndComments(); // Allow whitespace after comma before next key
        if (s.peek() == '}') throw s.error("Trailing comma in object is not allowed by strict JSON"); // Strict JSON
    }
    return obj;
}

} // namespace json
#endif // __JSON_H__