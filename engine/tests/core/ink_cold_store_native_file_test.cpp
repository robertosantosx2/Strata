// SPDX-License-Identifier: MIT
// Physical-file regression for InkColdStore's native .strata extent reader.
// This deliberately tests the PR-head contiguous descriptor API; it does not
// test GGUF/safetensors adapters or the .strata manifest parser.

#include "ink_cold_store.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path path = fs::temp_directory_path() /
        ("strata-ink-cold-store-native-" + std::to_string(stamp) + ".strata");

    // A non-sector-aligned absolute extent exercises head padding, aligned
    // physical reads, and reconstruction of the requested expert body.
    constexpr uint64_t file_off = 12345;
    constexpr uint32_t body_bytes = 8192;
    constexpr uint32_t dn_off = 4096;
    std::vector<uint8_t> file_bytes(65536);
    for (size_t i = 0; i < file_bytes.size(); ++i) {
        file_bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
    }

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "cannot create native ColdStore fixture: " << path << "\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(file_bytes.data()),
                  static_cast<std::streamsize>(file_bytes.size()));
        if (!out) {
            std::cerr << "cannot write native ColdStore fixture\n";
            return 1;
        }
    }

    ink2::InkColdExpertDesc desc{};
    desc.file_off = file_off;
    desc.body_bytes = body_bytes;
    desc.dn_off = dn_off;

    ink2::InkColdStore store;
    std::string err;
    if (!store.open(path.string(), {desc}, body_bytes, body_bytes * 16u, 2, err)) {
        std::cerr << "InkColdStore::open failed: " << err << "\n";
        fs::remove(path);
        return 1;
    }

    int slot = -1;
    const uint8_t* body = store.demand_wait(0, 0, 1, slot);
    if (!body || slot < 0) {
        std::cerr << "InkColdStore::demand_wait failed for physical extent\n";
        store.close();
        fs::remove(path);
        return 1;
    }

    bool matches = true;
    for (size_t i = 0; i < body_bytes; ++i) {
        if (body[i] != file_bytes[static_cast<size_t>(file_off) + i]) {
            std::cerr << "extent mismatch at body byte " << i
                      << ": expected "
                      << static_cast<unsigned>(file_bytes[static_cast<size_t>(file_off) + i])
                      << ", got " << static_cast<unsigned>(body[i]) << "\n";
            matches = false;
            break;
        }
    }

    store.unpin_slot(slot);
    store.close();
    std::error_code remove_error;
    fs::remove(path, remove_error);

    if (!matches) return 1;
    std::cout << "PASS: physical .strata extent read at absolute offset "
              << file_off << " (" << body_bytes << " bytes, dn_off " << dn_off
              << ", direct_io=" << "runtime-selected" << ")\n";
    return 0;
}
