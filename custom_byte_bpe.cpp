// Byte BPE implemented here: frequency counting, left-to-right merges,
// and ranked heap encoding. No external tokenizer implementation is used.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

using U32 = uint32_t;
static constexpr U32 NONE = std::numeric_limits<U32>::max();
static thread_local std::string error_message;

static U32* copy_output(const std::vector<U32>& values) {
    auto* result = static_cast<U32*>(std::malloc(std::max<size_t>(1, values.size()) * sizeof(U32)));
    if (!result) throw std::bad_alloc();
    if (!values.empty()) std::memcpy(result, values.data(), values.size() * sizeof(U32));
    return result;
}

extern "C" const char* bbpe_error() { return error_message.c_str(); }
extern "C" void bbpe_free(U32* values) { std::free(values); }

extern "C" int bbpe_train(const unsigned char* bytes, size_t length, U32 vocab_size,
                          U32** pairs_out, U32** ids_out, size_t* ids_length) {
    try {
        if (!length || vocab_size < 256 || vocab_size > 8192)
            throw std::runtime_error("Invalid corpus or vocabulary size");
        std::vector<U32> ids;
        ids.reserve(length + length / 100);
        for (size_t i = 0; i < length; ++i) {
            ids.push_back(bytes[i]);
            if (bytes[i] == '\n') ids.push_back(NONE); // Never cross line/article boundaries.
        }
        std::vector<U32> counts(static_cast<size_t>(vocab_size) * vocab_size, 0);
        std::vector<U32> touched, pairs;
        pairs.reserve((vocab_size - 256) * 2);
        for (U32 new_id = 256; new_id < vocab_size; ++new_id) {
            U32 best_key = NONE, best_count = 0;
            for (size_t i = 1; i < ids.size(); ++i) {
                U32 left = ids[i - 1], right = ids[i];
                if (left == NONE || right == NONE) continue;
                U32 key = left * vocab_size + right;
                if (counts[key] == 0) touched.push_back(key);
                U32 count = ++counts[key];
                if (count > best_count || (count == best_count && key < best_key)) {
                    best_count = count;
                    best_key = key;
                }
            }
            if (best_key == NONE) throw std::runtime_error("Corpus exhausted before target vocabulary");
            U32 left = best_key / vocab_size, right = best_key % vocab_size;
            pairs.push_back(left);
            pairs.push_back(right);
            size_t write = 0, read = 0;
            while (read < ids.size()) {
                if (read + 1 < ids.size() && ids[read] == left && ids[read + 1] == right) {
                    ids[write++] = new_id;
                    read += 2;
                } else {
                    ids[write++] = ids[read++];
                }
            }
            ids.resize(write);
            for (U32 key : touched) counts[key] = 0;
            touched.clear();
            if ((new_id + 1) % 256 == 0) {
                std::printf("Custom byte BPE: vocab=%u\n", new_id + 1);
                std::fflush(stdout);
            }
        }
        ids.erase(std::remove(ids.begin(), ids.end(), NONE), ids.end());
        *pairs_out = copy_output(pairs);
        *ids_out = copy_output(ids);
        *ids_length = ids.size();
        return 0;
    } catch (const std::exception& error) {
        error_message = error.what();
        return -1;
    }
}

struct Candidate {
    U32 rank;
    int position;
    bool operator<(const Candidate& other) const {
        return rank != other.rank ? rank > other.rank : position > other.position;
    }
};

extern "C" int bbpe_encode(const unsigned char* bytes, size_t length,
                           const U32* pairs, U32 merge_count,
                           U32** ids_out, size_t* ids_length) {
    try {
        U32 vocab_size = 256 + merge_count;
        std::vector<U32> ranks(static_cast<size_t>(vocab_size) * vocab_size, NONE);
        for (U32 rank = 0; rank < merge_count; ++rank)
            ranks[pairs[rank * 2] * vocab_size + pairs[rank * 2 + 1]] = rank;
        std::vector<U32> result;
        result.reserve(length);
        for (size_t begin = 0; begin < length;) {
            size_t end = begin;
            while (end < length && bytes[end] != '\n') ++end;
            if (end < length) ++end; // Include newline in this sequence.
            size_t n = end - begin;
            if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Encoding line too large");
            std::vector<U32> ids(n);
            std::vector<int> previous(n), next(n);
            std::priority_queue<Candidate> heap;
            for (size_t i = 0; i < n; ++i) {
                ids[i] = bytes[begin + i];
                previous[i] = static_cast<int>(i) - 1;
                next[i] = i + 1 < n ? static_cast<int>(i + 1) : -1;
            }
            auto offer = [&](int position) {
                if (position < 0 || ids[position] == NONE || next[position] < 0) return;
                U32 rank = ranks[ids[position] * vocab_size + ids[next[position]]];
                if (rank != NONE) heap.push({rank, position});
            };
            for (size_t i = 0; i + 1 < n; ++i) offer(static_cast<int>(i));
            while (!heap.empty()) {
                Candidate candidate = heap.top();
                heap.pop();
                int left = candidate.position;
                if (ids[left] == NONE || next[left] < 0) continue;
                int right = next[left];
                U32 rank = ranks[ids[left] * vocab_size + ids[right]];
                if (rank != candidate.rank) continue; // Stale candidate.
                ids[left] = 256 + rank;
                ids[right] = NONE;
                next[left] = next[right];
                if (next[left] >= 0) previous[next[left]] = left;
                offer(previous[left]);
                offer(left);
            }
            for (int i = 0; i >= 0; i = next[i]) result.push_back(ids[i]);
            begin = end;
        }
        *ids_out = copy_output(result);
        *ids_length = result.size();
        return 0;
    } catch (const std::exception& error) {
        error_message = error.what();
        return -1;
    }
}
