#include "ink_cold_store.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

int main() {
    std::vector<ink2::InkColdExpertDesc> descs(2);

    // Whole-expert source callback: this is the GGUF adapter/repack boundary used by demand reads.
    descs[0].file_off = 0;
    descs[0].body_bytes = 16;
    descs[0].n_spans = 1;
    descs[0].spans[0] = {-1, 0, 0, 16};

    // Three independent projections: prefix/header, gate/up and down payloads land in one ring slot.
    descs[1].file_off = 0;
    descs[1].body_bytes = 64;
    descs[1].n_spans = 3;
    for (size_t i = 0; i < descs[1].prefix.size(); ++i)
        descs[1].prefix[i] = static_cast<uint8_t>(i);
    descs[1].spans[0] = {0, 100, 32, 8};
    descs[1].spans[1] = {1, 200, 40, 8};
    descs[1].spans[2] = {2, 300, 48, 8};

    ink2::InkColdStore store;
    std::string err;
    auto alloc = [](size_t n) -> uint8_t* { return static_cast<uint8_t*>(std::malloc(n)); };
    auto free_mem = [](uint8_t* p) { std::free(p); };
    auto reader = [](int file_idx, uint64_t file_off, uint8_t* dst, size_t len) -> bool {
        if (file_idx == -1 && file_off == 0 && len == 16) {
            for (size_t i = 0; i < len; ++i) dst[i] = static_cast<uint8_t>(0x40 + i);
            return true;
        }
        if (file_idx == 0 && file_off == 100 && len == 8) {
            std::memset(dst, 0xA0, len);
            return true;
        }
        if (file_idx == 1 && file_off == 200 && len == 8) {
            std::memset(dst, 0xB1, len);
            return true;
        }
        if (file_idx == 2 && file_off == 300 && len == 8) {
            std::memset(dst, 0xC2, len);
            return true;
        }
        return false;
    };

    assert(store.open("unused-in-source-reader-mode", std::move(descs), 64, 256u * 1024u,
                      2, err, alloc, free_mem, reader));

    int slot0 = -1;
    const uint8_t* body0 = store.demand_wait(0, 0, 1, slot0);
    assert(body0 && slot0 >= 0);
    for (size_t i = 0; i < 16; ++i)
        assert(body0[i] == static_cast<uint8_t>(0x40 + i));
    store.unpin_slot(slot0);

    int slot1 = -1;
    const uint8_t* body1 = store.demand_wait(1, 0, 1, slot1);
    assert(body1 && slot1 >= 0);
    for (size_t i = 0; i < 32; ++i) assert(body1[i] == static_cast<uint8_t>(i));
    for (size_t i = 32; i < 40; ++i) assert(body1[i] == 0xA0);
    for (size_t i = 40; i < 48; ++i) assert(body1[i] == 0xB1);
    for (size_t i = 48; i < 56; ++i) assert(body1[i] == 0xC2);
    for (size_t i = 56; i < 64; ++i) assert(body1[i] == 0);
    store.unpin_slot(slot1);

    store.close();
    return 0;
}
