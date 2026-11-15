#ifndef __SAFETENSOR_H__
#define __SAFETENSOR_H__

#include "common/common.h"
#include "utils/utils.h"
#include "utils/json.h"
#include "common/torch_common.h"

#include <array>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <future>         // For std::async and std::future
#include <thread>         // For std::thread::hardware_concurrency
#include <algorithm>      // For std::transform, std::all_of
#include <numeric>        // For std::accumulate
#include <memory>         // For std::shared_ptr
#include <optional>       // For std::optional (a dtype that may be left as stored)

namespace safetensor
{
    // Helper struct for DType information
    struct DTypeInfo {
        torch::ScalarType torch_type;
        size_t element_size;
        std::string safetensor_name; // For reverse mapping if needed
    };

    // Centralized DType mapping
    inline const std::unordered_map<std::string, DTypeInfo>& get_safetensor_dtype_map() {
        static const std::unordered_map<std::string, DTypeInfo> dtype_map = {
            {"F64",  {torch::kFloat64, sizeof(double),   "F64"}},
            {"F32",  {torch::kFloat32, sizeof(float),    "F32"}},
            {"F16",  {torch::kHalf,    2,                "F16"}}, // torch::Half is 2 bytes
            {"BF16", {torch::kBFloat16,2,                "BF16"}},// torch::BFloat16 is 2 bytes
            {"I64",  {torch::kInt64,   sizeof(int64_t),  "I64"}},
            {"I32",  {torch::kInt32,   sizeof(int32_t),  "I32"}},
            {"I16",  {torch::kInt16,   sizeof(int16_t),  "I16"}},
            {"I8",   {torch::kInt8,    sizeof(int8_t),   "I8"}},
            {"U8",   {torch::kUInt8,   sizeof(uint8_t),  "U8"}},
            {"U16",  {torch::ScalarType::UInt16, sizeof(uint16_t), "U16"}},
            {"U32",  {torch::ScalarType::UInt32, sizeof(uint32_t), "U32"}},
            {"U64",  {torch::ScalarType::UInt64, sizeof(uint64_t), "U64"}},
            {"BOOL", {torch::kBool,    sizeof(bool),     "BOOL"}} // Typically 1 byte in memory for torch
        };
        return dtype_map;
    }

    // 바이트 순서(엔디안) 처리 함수 추가
    inline int64_t convert_endianness(int64_t value) {
        // 실행 환경이 리틀 엔디안인지 확인
        static const int endian_check = 1;
        static const bool is_little_endian = (*reinterpret_cast<const char*>(&endian_check) == 1);
        
        // 리틀 엔디안 환경이면 변환 필요 없음
        if (is_little_endian) return value;
        
        // 빅 엔디안 환경이면 바이트 순서 변환
        return ((value & 0xFF) << 56) | 
               ((value & 0xFF00) << 40) | 
               ((value & 0xFF0000) << 24) | 
               ((value & 0xFF000000) << 8) | 
               ((value >> 8) & 0xFF000000) | 
               ((value >> 24) & 0xFF0000) | 
               ((value >> 40) & 0xFF00) | 
               ((value >> 56) & 0xFF);
    }

    // --- Main Safetensor Class ---
    class Safetensor
    {
        std::string _filename;
        int64_t _header_size_bytes;
        json::Json _parsed_header;
        
        // 메모리 매핑 고려 (옵션)
        // std::unique_ptr<MemoryMappedFile> _mmapped_file;
        
        // 읽기 전용으로 사용되는 파일은 한 번만 열고 캐싱하는 것을 고려
        // std::shared_ptr<std::ifstream> _cached_file;

        // 텐서 로딩 함수 개선
        // A contiguous piece of the file: where it starts and how many bytes it has.
        struct ByteRun {
            int64_t offset;
            size_t nbytes;
        };

        // Reads the runs, in order, one after another into the tensor's memory: together they are its bytes.
        static void read_tensor_data_from_file(
            const std::string& filename,
            const std::vector<ByteRun>& runs,
            torch::Tensor& target_tensor
        ) {
            // Unbuffered: a run goes from the file straight into the tensor, and nothing beyond the run is read.
            std::ifstream file;
            file.rdbuf()->pubsetbuf(nullptr, 0);
            file.open(filename, std::ios::in | std::ios::binary);
            if (!file.is_open()) { 
                throw std::runtime_error("Task: Failed to open file for reading tensor data: " + filename); 
            }
            
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();

            char* target = reinterpret_cast<char*>(target_tensor.data_ptr());
            const size_t target_bytes = target_tensor.nbytes();
            size_t bytes_read = 0;
            for (const ByteRun& run : runs) {
                // 경계 확인 - 읽으려는 바이트가 파일 끝을 넘어가지 않는지
                if (run.offset < 0 || static_cast<size_t>(run.offset) + run.nbytes > file_size) {
                    throw std::runtime_error("Task: Reading beyond end of file. Offset " + std::to_string(run.offset) + 
                                            " + size " + std::to_string(run.nbytes) + 
                                            " exceeds file size " + std::to_string(file_size));
                }
                if (bytes_read + run.nbytes > target_bytes) {
                    throw std::runtime_error("Task: Reading more than the tensor holds (" + 
                                            std::to_string(target_bytes) + " bytes)");
                }

                file.seekg(run.offset, std::ios::beg);
                if (file.fail()) { 
                    throw std::runtime_error("Task: Failed to seek in file to offset " + std::to_string(run.offset)); 
                }

                file.read(target + bytes_read, static_cast<std::streamsize>(run.nbytes));
                if (static_cast<size_t>(file.gcount()) != run.nbytes) {
                    throw std::runtime_error("Task: Failed to read all tensor data. Expected " + 
                                             std::to_string(run.nbytes) + " bytes, got " + 
                                             std::to_string(file.gcount()));
                }
                bytes_read += run.nbytes;
            }
            if (bytes_read != target_bytes) {
                throw std::runtime_error("Task: Read " + std::to_string(bytes_read) + " bytes for a tensor of " + 
                                         std::to_string(target_bytes) + " bytes");
            }
            file.close();
        }

    public:
        struct TensorInfo {
            std::string name;
            DTypeInfo dtype_info;
            std::vector<int64_t> shape;
            int64_t offset_begin;
            int64_t offset_end;
            // offset_begin and offset_end are as the header gives them, from the start of the data section;
            // file_offset is where the tensor's first byte is in the file.
            int64_t file_offset;
            
            // 요소 개수와 바이트 크기를 계산하는 유틸리티 함수 추가
            size_t num_elements() const {
                return std::accumulate(shape.begin(), shape.end(), 
                                      size_t(1), std::multiplies<size_t>());
            }
            
            size_t total_bytes() const {
                return num_elements() * dtype_info.element_size;
            }
        };

        Safetensor(const std::string& filename) : _filename(filename) {
            std::ifstream file(filename, std::ios::in | std::ios::binary);
            if (!file.is_open()) { 
                throw std::runtime_error("Failed to open Safetensor file: " + filename); 
            }

            // 1. 8바이트 리틀 엔디안 정수 읽기 - 헤더 길이용
            file.read(reinterpret_cast<char*>(&_header_size_bytes), sizeof(_header_size_bytes));
            if (file.gcount() != sizeof(_header_size_bytes)) { 
                throw std::runtime_error("Failed to read header size from file: " + filename); 
            }
            
            // 엔디안 변환 적용
            _header_size_bytes = convert_endianness(_header_size_bytes);

            // 헤더 크기의 유효성 검사
            if (_header_size_bytes <= 0 || _header_size_bytes > 256 * 1024 * 1024) {
                 throw std::runtime_error("Invalid or excessively large header size: " + 
                                         std::to_string(_header_size_bytes)); 
            }
            
            // 헤더 JSON 문자열 읽기
            std::vector<char> header_buffer(_header_size_bytes);
            file.seekg(8, std::ios::beg);
            if (!file.good()) { 
                throw std::runtime_error("Failed to seek to header position in file: " + filename); 
            }
            
            file.read(header_buffer.data(), _header_size_bytes);
            if (file.gcount() != _header_size_bytes) {
                throw std::runtime_error("Failed to read complete header. Expected " + 
                                        std::to_string(_header_size_bytes) + " bytes, got " + 
                                        std::to_string(file.gcount()));
            }
            
            try {
                _parsed_header = json::Json(std::string(header_buffer.begin(), header_buffer.end()));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse header JSON: " + std::string(e.what()));
            }
            
            file.close();
        }

        // 사용 가능한 모든 텐서 이름 목록 반환
        std::vector<std::string> get_tensor_names() const {
            std::vector<std::string> names;
            for(const auto& [name, _] : _parsed_header.get<json::JsonObject>()) {
                // "__metadata__" is the file's own string map, not a tensor.
                if (name == "__metadata__") continue;
                names.push_back(name);
            }
            return names;
        }

        bool has_tensor(const std::string& name) const {
            return name != "__metadata__" && _parsed_header.hasKey(name);
        }

        const std::string& filename() const { return _filename; }

        // 텐서 메타데이터 반환 - 잘 구현됨
        TensorInfo get_tensor_info(const std::string& name) const {
            // 검사 추가: 헤더에 해당 이름의 텐서가 있는지 확인
            if (!has_tensor(name)) {
                throw std::runtime_error("Tensor with name '" + name + "' not found in safetensor file.");
            }
            
            const json::JsonData& _tensor_info = _parsed_header.at(name);
            if (_tensor_info.getType() != json::JsonDataType::JSON_OBJECT) {
                throw std::runtime_error("Tensor info for '" + name + "' is not an object.");
            }

            TensorInfo tensor_info;
            tensor_info.name = name;
            
            // 필수 필드 존재 확인
            if (!_tensor_info.hasKey("dtype") || !_tensor_info.hasKey("shape") || 
                !_tensor_info.hasKey("data_offsets")) {
                throw std::runtime_error("Tensor '" + name + "' missing required metadata fields.");
            }
            
            const std::string& dtype_name = _tensor_info.at("dtype").get<std::string>();
            const auto& dtype_map = get_safetensor_dtype_map();
            const auto dtype_entry = dtype_map.find(dtype_name);
            if (dtype_entry == dtype_map.end()) {
                throw std::runtime_error("Tensor '" + name + "' has dtype '" + dtype_name + 
                                        "', which is not read here.");
            }
            tensor_info.dtype_info = dtype_entry->second;
            
            // 모양 정보 처리 (a JSON integer is a long long, which int64_t is not everywhere)
            const auto& _shape = _tensor_info.at("shape").get<json::JsonArray>();
            tensor_info.shape = std::vector<int64_t>(_shape.size());
            std::transform(_shape.begin(), _shape.end(), tensor_info.shape.begin(),
                [](const json::JsonData& val) { return static_cast<int64_t>(val.get<long long>()); });
            if (std::any_of(tensor_info.shape.begin(), tensor_info.shape.end(),
                            [](int64_t size) { return size < 0; })) {
                throw std::runtime_error("Negative size in the shape of tensor '" + name + "'");
            }
            
            // 오프셋 정보 처리
            const auto& _offsets = _tensor_info.at("data_offsets").get<json::JsonArray>();
            if (_offsets.size() != 2) {
                throw std::runtime_error("Invalid data_offsets for tensor '" + name + "'");
            }
            
            tensor_info.offset_begin = _offsets[0].get<long long>();
            tensor_info.offset_end = _offsets[1].get<long long>();
            // The data section starts after the 8-byte header length and the header itself.
            tensor_info.file_offset = 8 + _header_size_bytes + tensor_info.offset_begin;
            
            // 오프셋 유효성 검사 (a tensor without elements has begin == end)
            if (tensor_info.offset_begin < 0 || tensor_info.offset_begin > tensor_info.offset_end) {
                throw std::runtime_error("Invalid offset range for tensor '" + name + "'");
            }
            
            // 예상 크기와 실제 오프셋 범위 일치 확인
            size_t expected_bytes = tensor_info.total_bytes();
            size_t actual_bytes = tensor_info.offset_end - tensor_info.offset_begin;
            if (expected_bytes != actual_bytes) {
                throw std::runtime_error("Size mismatch for tensor '" + name + 
                                        "'. Expected " + std::to_string(expected_bytes) + 
                                        " bytes, but offset range indicates " + std::to_string(actual_bytes));
            }
            
            return tensor_info;
        }

        // A half-open range [begin, end) along one dimension (step 1).
        struct Range {
            int64_t begin;
            int64_t end;
        };

    private:
        // A range for every dimension of the tensor: the dimensions `ranges` does not reach are taken whole.
        static std::vector<Range> complete_ranges(const TensorInfo& tensor_info, const std::vector<Range>& ranges) {
            const size_t rank = tensor_info.shape.size();
            if (ranges.size() > rank) {
                throw std::invalid_argument("Tensor '" + tensor_info.name + "' has " + std::to_string(rank) + 
                                            " dimensions; " + std::to_string(ranges.size()) + " ranges were given.");
            }
            std::vector<Range> block(rank);
            for (size_t d = 0; d < rank; ++d) {
                block[d] = d < ranges.size() ? ranges[d] : Range{0, tensor_info.shape[d]};
                if (block[d].begin < 0 || block[d].begin > block[d].end || block[d].end > tensor_info.shape[d]) {
                    throw std::invalid_argument("Range [" + std::to_string(block[d].begin) + ", " + 
                                                std::to_string(block[d].end) + ") is outside dimension " + 
                                                std::to_string(d) + " (size " + std::to_string(tensor_info.shape[d]) + 
                                                ") of tensor '" + tensor_info.name + "'.");
                }
            }
            return block;
        }

        // The pieces of the file a block of the tensor is made of, in the order they fill the block.
        // The tensor is stored in row-major order, so the last dimension the block cuts lies in the file as one
        // piece together with all the dimensions after it: a block of whole rows (only the first dimension cut)
        // is a single piece, and so is the whole tensor.
        static std::vector<ByteRun> block_runs(const TensorInfo& tensor_info, const std::vector<Range>& block) {
            const size_t rank = block.size();
            const int64_t element_size = static_cast<int64_t>(tensor_info.dtype_info.element_size);
            std::vector<ByteRun> runs;
            for (const Range& range : block) {
                if (range.begin == range.end) return runs;  // a block without elements
            }

            // elements[d]: the elements under one index of dimension d - 1 (elements[rank] = 1)
            std::vector<int64_t> elements(rank + 1, 1);
            for (size_t d = rank; d-- > 0;) elements[d] = elements[d + 1] * tensor_info.shape[d];

            // cut: one past the last dimension the block does not take whole
            size_t cut = rank;
            while (cut > 0 && block[cut - 1].begin == 0 && block[cut - 1].end == tensor_info.shape[cut - 1]) --cut;
            if (cut == 0) {
                runs.push_back({tensor_info.file_offset, tensor_info.total_bytes()});
                return runs;
            }

            const size_t last = cut - 1;
            const size_t run_bytes = static_cast<size_t>(
                (block[last].end - block[last].begin) * elements[cut] * element_size);
            size_t run_count = 1;
            std::vector<int64_t> index(last);
            for (size_t d = 0; d < last; ++d) {
                index[d] = block[d].begin;
                run_count *= static_cast<size_t>(block[d].end - block[d].begin);
            }
            runs.reserve(run_count);
            for (;;) {
                int64_t first = block[last].begin * elements[cut];
                for (size_t d = 0; d < last; ++d) first += index[d] * elements[d + 1];
                runs.push_back({tensor_info.file_offset + first * element_size, run_bytes});

                // the next index of the dimensions before `last`, the later ones turning faster
                size_t d = last;
                for (;;) {
                    if (d == 0) return runs;
                    --d;
                    if (++index[d] < block[d].end) break;
                    index[d] = block[d].begin;
                }
            }
        }

    public:
        // 코어 텐서 로딩 로직 - 동기/비동기 사용 가능
        // ranges: the block of the tensor to read (none: the whole tensor); only the block's bytes are read.
        // target_scalar_type: none keeps the dtype the file stores.
        static torch::Tensor load_and_convert_tensor(
            const std::string& filename_for_task,
            const TensorInfo& tensor_info,
            const std::vector<Range>& ranges,
            std::optional<torch::ScalarType> target_scalar_type,
            const torch::Device& target_device
        ) {
            // 텐서 형태 준비
            const std::vector<Range> block = complete_ranges(tensor_info, ranges);
            std::vector<int64_t> shape(block.size());
            std::transform(block.begin(), block.end(), shape.begin(),
                [](const Range& range) { return range.end - range.begin; });

            // CPU에 원본 데이터타입으로 텐서 생성
            auto options_original_cpu = torch::TensorOptions()
                                            .dtype(tensor_info.dtype_info.torch_type)
                                            .device(torch::kCPU);
            torch::Tensor cpu_tensor = torch::empty(shape, options_original_cpu);
            
            // 파일에서 데이터 읽기
            try {
                read_tensor_data_from_file(filename_for_task, block_runs(tensor_info, block), cpu_tensor);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load tensor '" + tensor_info.name + 
                                        "': " + std::string(e.what()));
            }
            
            // 타입 변환 및 디바이스 이동
            const torch::ScalarType scalar_type = target_scalar_type.value_or(cpu_tensor.scalar_type());
            if (cpu_tensor.scalar_type() == scalar_type && cpu_tensor.device() == target_device) {
                return cpu_tensor;
            } else {
                // CUDA로 이동하는 경우 non_blocking 활성화 고려
                bool non_blocking = target_device.is_cuda();
                return cpu_tensor.to(target_device, scalar_type, non_blocking, /*copy=*/false);
            }
        }

        // The whole tensor, with the target fixed at compile time (float32 on the CPU unless named).
        // 대상 디바이스 설정: CUDA without an index is the CUDA device in use.
        template<torch::ScalarType TargetScalarType = torch::kFloat32, 
                 torch::DeviceType TargetDeviceType = torch::kCPU, 
                 int16_t TargetDeviceIndex = -1>
        static torch::Tensor load_and_convert_tensor(
            const std::string& filename_for_task,
            const TensorInfo& tensor_info
        ) {
            return load_and_convert_tensor(filename_for_task, tensor_info, {}, TargetScalarType,
                                           torch::Device(TargetDeviceType, TargetDeviceIndex));
        }

        // 동기식 텐서 읽기 - a block of the tensor, dtype and device given at run time
        torch::Tensor read(
            const std::string& name,
            const std::vector<Range>& ranges = {},
            std::optional<torch::ScalarType> target_scalar_type = std::nullopt,
            const torch::Device& target_device = torch::Device(torch::kCPU)
        ) const {
            return load_and_convert_tensor(_filename, get_tensor_info(name), ranges, target_scalar_type, target_device);
        }

        // 동기식 텐서 읽기
        template<torch::ScalarType TargetScalarType = torch::kFloat32, 
                 torch::DeviceType TargetDeviceType = torch::kCPU, 
                 int16_t TargetDeviceIndex = -1>
        torch::Tensor readTensor(const std::string& name) {
            const TensorInfo tensor_info = get_tensor_info(name);
            return load_and_convert_tensor<TargetScalarType, TargetDeviceType, TargetDeviceIndex>(
                _filename, tensor_info);
        }

        // 비동기식 텐서 읽기 - 스레드 풀 관리 개선
        template<torch::ScalarType TargetScalarType = torch::kFloat32, 
                 torch::DeviceType TargetDeviceType = torch::kCPU, 
                 int16_t TargetDeviceIndex = -1>
        std::future<torch::Tensor> readTensorAsync(const std::string& name) {
            TensorInfo tensor_info = get_tensor_info(name); // 복사본 생성
            std::string current_filename = _filename; // 파일명 복사
            
            // 비동기 실행
            return std::async(std::launch::async, 
                [current_filename, tensor_info]() -> torch::Tensor {
                    return load_and_convert_tensor<TargetScalarType, TargetDeviceType, TargetDeviceIndex>(
                        current_filename, tensor_info);
                });
        }

        // 슬라이싱 - 최적화된 구현
        template<torch::ScalarType TargetScalarType = torch::kFloat32,
                 torch::DeviceType TargetDeviceType = torch::kCPU,
                 int16_t TargetDeviceIndex = -1>
        torch::Tensor getSlice(const std::string& name, const std::vector<torch::indexing::TensorIndex>& slice_indices) {
            TensorInfo tensor_info = get_tensor_info(name);
            
            // 슬라이스 인덱스 개수가 텐서 차원과 일치하는지 확인
            if (slice_indices.size() != tensor_info.shape.size()) {
                throw std::runtime_error("Number of slice indices (" + std::to_string(slice_indices.size()) + 
                                        ") doesn't match tensor dimensions (" + std::to_string(tensor_info.shape.size()) + ")");
            }
            
            // 슬라이스 인덱스 분석하여 정확한 바이트 범위 결정: integers and slices of step 1 name a block, which
            // is read alone (read()); the dimension of an integer is then dropped, as indexing drops it.
            std::vector<Range> ranges;
            std::vector<int64_t> indexed_shape;
            bool is_block = true;
            for (size_t d = 0; d < slice_indices.size() && is_block; ++d) {
                const torch::indexing::TensorIndex& index = slice_indices[d];
                const int64_t size = tensor_info.shape[d];
                if (index.is_integer()) {
                    int64_t i = c10::SymInt(index.integer()).expect_int();
                    if (i < 0) i += size;
                    if (i < 0 || i >= size) {
                        throw std::out_of_range("Index out of range for dimension " + std::to_string(d) + 
                                                " (size " + std::to_string(size) + ") of tensor '" + name + "'");
                    }
                    ranges.push_back({i, i + 1});
                } else if (index.is_slice() && c10::SymInt(index.slice().step()).expect_int() == 1) {
                    int64_t begin = c10::SymInt(index.slice().start()).expect_int();
                    int64_t end = c10::SymInt(index.slice().stop()).expect_int();
                    begin = std::clamp<int64_t>(begin < 0 ? begin + size : begin, 0, size);
                    end = std::clamp<int64_t>(end < 0 ? end + size : end, begin, size);
                    ranges.push_back({begin, end});
                    indexed_shape.push_back(end - begin);
                } else {
                    is_block = false;
                }
            }
            if (is_block) {
                return read(name, ranges, TargetScalarType, torch::Device(TargetDeviceType, TargetDeviceIndex))
                           .reshape(indexed_shape);
            }
            
            // Any other index (None, an ellipsis, a tensor, a step): 전체 텐서를 로드한 후 메모리에서 슬라이싱
            torch::Tensor full_tensor = this->readTensor<TargetScalarType, TargetDeviceType, TargetDeviceIndex>(name);
            return full_tensor.index(slice_indices);
        }
        
        // 메모리 효율성 향상을 위한 추가 메서드:
        
        // 1. 여러 텐서를 한 번에 로드 (동기식)
        template<torch::ScalarType TargetScalarType = torch::kFloat32,
                 torch::DeviceType TargetDeviceType = torch::kCPU,
                 int16_t TargetDeviceIndex = -1>
        std::unordered_map<std::string, torch::Tensor> readTensors(const std::vector<std::string>& names) {
            std::unordered_map<std::string, torch::Tensor> result;
            for (const auto& name : names) {
                result[name] = readTensor<TargetScalarType, TargetDeviceType, TargetDeviceIndex>(name);
            }
            return result;
        }
        
        // 2. 여러 텐서를 한 번에 비동기 로드
        template<torch::ScalarType TargetScalarType = torch::kFloat32,
                 torch::DeviceType TargetDeviceType = torch::kCPU,
                 int16_t TargetDeviceIndex = -1>
        std::unordered_map<std::string, std::future<torch::Tensor>> readTensorsAsync(
            const std::vector<std::string>& names) {
            std::unordered_map<std::string, std::future<torch::Tensor>> futures;
            for (const auto& name : names) {
                futures[name] = readTensorAsync<TargetScalarType, TargetDeviceType, TargetDeviceIndex>(name);
            }
            return futures;
        }
    }; // class Safetensor
    // Regarding the original `SlicedSafetensor` struct:
    //
    // template<int _Dim>
    // struct SlicedSafetensor { ... }
    //
    // Challenges with `SlicedSafetensor` as presented:
    // 1. `_Dim` template parameter: Limits it to fixed-dimension tensors, while tensors can have dynamic rank.
    // 2. `index(std::array<size_t, _Dim> indices)`:
    //    - Offset calculation `offset += indices[i] * sizeof(float)` is incorrect for multi-dimensional arrays.
    //      It needs to use strides: `offset_in_elements = sum(indices[j] * strides[j])`.
    //    - Reading `tensor.numel() * sizeof(float)` reads the *entire original shape's data* from the new offset,
    //      which is not what indexing a single element or small slice means. It should read only the data
    //      for the requested element/sub-slice.
    //    - Re-opening the file (`_file(filename, ...)` in constructor and potentially in `index`) for each slice
    //      or each index operation is inefficient.
    //
    // A more viable `SlicedSafetensor` (or a "TensorView" on disk) would:
    // - Store a reference/shared_ptr to the main `Safetensor` object or its file handle and header.
    // - Define the slice it represents (e.g., using start offsets and shapes for the slice).
    // - Its `read_data()` or `get_sub_slice()` methods would then perform optimized reads for *that specific slice*,
    //   potentially by delegating to a more advanced `Safetensor::getSliceInternal` method.
    //
    // Given the complexity, using `Safetensor::getSlice` with `torch::indexing::TensorIndex` (which delegates
    // slicing to PyTorch after loading a relevant chunk) is a more practical starting point.

} // namespace safetensor
#endif // __SAFETENSOR_H__