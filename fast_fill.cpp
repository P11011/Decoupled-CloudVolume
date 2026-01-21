#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sys/mman.h>

extern "C" {

//g++ -O3 -mavx2 -fopenmp -fPIC -shared fast_fill.cpp -o libfastfill.so

// ==========================================
//  Uint8 填充 (原有逻辑)
// ==========================================
void parallel_fill_u8(unsigned char* data, size_t size_bytes, unsigned char value, int num_threads) {
    // 开启大页建议
    madvise(data, size_bytes, MADV_HUGEPAGE);

    if (size_bytes < 2 * 1024 * 1024) {
        std::memset(data, value, size_bytes);
        return;
    }

    __m256i v = _mm256_set1_epi8(value);
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk_size = size_bytes / nthreads;
        size_t start_idx = tid * chunk_size;
        size_t end_idx = (tid == nthreads - 1) ? size_bytes : start_idx + chunk_size;
        
        unsigned char* ptr = data + start_idx;
        size_t len = end_idx - start_idx;

        // 32字节对齐处理
        size_t align_offset = (32 - (reinterpret_cast<uintptr_t>(ptr) & 31)) & 31;
        align_offset = std::min(align_offset, len);
        if (align_offset > 0) {
            std::memset(ptr, value, align_offset);
            ptr += align_offset;
            len -= align_offset;
        }

        // 核心循环
        size_t block32_cnt = len / 32;
        size_t i = 0;
        // 预取距离
        const int PREFETCH_DIST = 256; 

        for (; i + 4 <= block32_cnt; i += 4) {
            _mm_prefetch(reinterpret_cast<const char*>(ptr + i * 32 + PREFETCH_DIST), _MM_HINT_T0);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + (i + 0) * 32), v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + (i + 1) * 32), v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + (i + 2) * 32), v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + (i + 3) * 32), v);
        }
        for (; i < block32_cnt; ++i) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i * 32), v);
        }
        size_t tail = len % 32;
        if (tail > 0) std::memset(ptr + block32_cnt * 32, value, tail);
    }
}

// ==========================================
//  Uint64 填充 (新增逻辑)
// ==========================================
void parallel_fill_u64(uint64_t* data, size_t num_elements, uint64_t value, int num_threads) {
    size_t size_bytes = num_elements * sizeof(uint64_t);
    madvise(data, size_bytes, MADV_HUGEPAGE);

    // 设置 AVX2 寄存器 (4个 64位整数)
    __m256i v = _mm256_set1_epi64x(value);
    
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t chunk_elems = num_elements / nthreads;
        size_t start_idx = tid * chunk_elems;
        size_t end_idx = (tid == nthreads - 1) ? num_elements : start_idx + chunk_elems;

        uint64_t* ptr = data + start_idx;
        size_t count = end_idx - start_idx;

        // 对齐处理 (AVX2 需要 32 字节对齐，即 4 个 uint64)
        // 检查当前地址是否 32 字节对齐
        size_t align_bytes = (32 - (reinterpret_cast<uintptr_t>(ptr) & 31)) & 31;
        size_t align_elems = align_bytes / sizeof(uint64_t);
        
        if (align_elems > 0 && count >= align_elems) {
            for (size_t k = 0; k < align_elems; ++k) ptr[k] = value;
            ptr += align_elems;
            count -= align_elems;
        }

        // 主循环：每次处理 4 个 __m256i，即 16 个 uint64 (128 字节)
        size_t i = 0;
        const int PREFETCH_DIST_BYTES = 256;

        for (; i + 16 <= count; i += 16) {
            _mm_prefetch(reinterpret_cast<const char*>(ptr + i + (PREFETCH_DIST_BYTES/8)), _MM_HINT_T0);
            
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i + 0), v);  // 4 elems
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i + 4), v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i + 8), v);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i + 12), v);
        }
        
        // 处理剩余的 4 元素块 (32字节)
        for (; i + 4 <= count; i += 4) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i), v);
        }

        // 处理尾部 (Standard Store)
        for (; i < count; ++i) {
            ptr[i] = value;
        }
    }
}

}