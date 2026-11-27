// SPDX-License-Identifier: MIT
// End-to-end native .strata v1.0 fixture: header -> manifest/directory -> section 4 -> ColdStore.

#include "strata/artifact/strata_reader.hpp"
#include "ink_cold_store.hpp"

#include <zstd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kPage = 4096;
uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(kPage - 1); }

struct Section {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::array<uint8_t, 32> sha{};
    uint32_t id = 0;
    uint32_t flags = 0;
    uint64_t reserved = 0;
};
static_assert(sizeof(Section) == 64);

bool write_all(std::fstream& f, uint64_t offset, const void* data, size_t bytes) {
    f.seekp(static_cast<std::streamoff>(offset));
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
}
}

int main() {
    namespace fs = std::filesystem;
    using strata::artifact::StrataExpertEntry;
    using strata::artifact::StrataFileReader;

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path path = fs::temp_directory_path() /
        ("strata-native-v1-coldstore-" + std::to_string(stamp) + ".strata");

    const std::string manifest = R"({"format":"strata-v1.0","fixture":true,"expert_count":1})";
    std::vector<uint8_t> compressed(ZSTD_compressBound(manifest.size()));
    const size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed.size(), manifest.data(), manifest.size(), 1);
    if (ZSTD_isError(compressed_size)) {
        std::cerr << "ZSTD_compress failed: " << ZSTD_getErrorName(compressed_size) << "\n";
        return 1;
    }
    compressed.resize(compressed_size);

    constexpr uint32_t body_bytes = 8192;
    constexpr uint32_t dn_off = 4096;
    constexpr uint64_t body_rel_off = 4096;
    std::vector<uint8_t> body(body_bytes);
    for (size_t i = 0; i < body.size(); ++i)
        body[i] = static_cast<uint8_t>((i * 29u + 17u) & 0xffu);

    StrataExpertEntry expert{};
    expert.expert_id = 7;
    expert.gu_code_id = 3;
    expert.dn_code_id = 3;
    expert.body_offset = body_rel_off;
    expert.body_bytes = body_bytes;
    expert.dn_offset = dn_off;

    std::array<Section, 6> sections{};
    uint64_t cursor = kPage;
    sections[0].offset = cursor;
    sections[0].bytes = compressed.size();
    sections[0].id = 0;
    cursor = align_page(cursor + compressed.size());

    sections[3].offset = cursor;
    sections[3].bytes = sizeof(expert);
    sections[3].id = 3;
    cursor = align_page(cursor + sizeof(expert));

    sections[4].offset = cursor;
    sections[4].bytes = body_rel_off + body.size();
    sections[4].id = 4;
    cursor = align_page(cursor + sections[4].bytes);

    {
        std::fstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!f) {
            std::cerr << "cannot create fixture " << path << "\n";
            return 1;
        }
        std::vector<uint8_t> zero(static_cast<size_t>(cursor), 0);
        f.write(reinterpret_cast<const char*>(zero.data()), static_cast<std::streamsize>(zero.size()));
        if (!f) return 1;

        if (!write_all(f, sections[0].offset, compressed.data(), compressed.size()) ||
            !write_all(f, sections[3].offset, &expert, sizeof(expert)) ||
            !write_all(f, sections[4].offset + body_rel_off, body.data(), body.size())) {
            std::cerr << "failed writing fixture sections\n";
            return 1;
        }

        std::array<uint8_t, kPage> header{};
        const uint8_t magic[8] = {'S','T','R','A','T','A',0,1};
        std::memcpy(header.data(), magic, sizeof(magic));
        const uint32_t flags = 0;
        const uint16_t profile = 0;
        const uint16_t num_sections = 6;
        std::memcpy(header.data() + 8, &flags, sizeof(flags));
        std::memcpy(header.data() + 12, &profile, sizeof(profile));
        std::memcpy(header.data() + 14, &num_sections, sizeof(num_sections));
        for (size_t i = 0; i < sections.size(); ++i)
            std::memcpy(header.data() + 0x10 + i * sizeof(Section), &sections[i], sizeof(Section));
        if (!write_all(f, 0, header.data(), header.size())) return 1;
    }

    std::string err;
    StrataFileReader reader;
    if (!reader.open(path.string(), err)) {
        std::cerr << "StrataFileReader::open failed: " << err << "\n";
        fs::remove(path);
        return 1;
    }
    if (reader.manifest_json() != manifest || reader.expert_directory().size() != 1) {
        std::cerr << "manifest or expert directory did not parse as expected\n";
        fs::remove(path);
        return 1;
    }

    std::vector<uint8_t> reader_body;
    if (!reader.read_expert_body(0, reader_body, err) || reader_body != body) {
        std::cerr << "StrataFileReader::read_expert_body failed/mismatched: " << err << "\n";
        fs::remove(path);
        return 1;
    }

    const auto& section4 = reader.header().sections[4];
    const auto& parsed = reader.expert_directory()[0];
    ink2::InkColdExpertDesc desc{};
    desc.file_off = section4.file_offset + parsed.body_offset;
    desc.body_bytes = parsed.body_bytes;
    desc.dn_off = parsed.dn_offset;
    desc.gu_code = parsed.gu_code_id;
    desc.dn_code = parsed.dn_code_id;

    ink2::InkColdStore cold;
    if (!cold.open(path.string(), {desc}, body_bytes, body_bytes * 16u, 2, err)) {
        std::cerr << "InkColdStore::open failed: " << err << "\n";
        fs::remove(path);
        return 1;
    }
    int slot = -1;
    const uint8_t* loaded = cold.demand_wait(0, 0, 1, slot);
    bool matches = loaded != nullptr && slot >= 0;
    if (matches) {
        for (size_t i = 0; i < body.size(); ++i) {
            if (loaded[i] != body[i]) { matches = false; break; }
        }
        cold.unpin_slot(slot);
    }
    cold.close();
    reader.close();
    std::error_code ec;
    fs::remove(path, ec);

    if (!matches) {
        std::cerr << "ColdStore body did not match the expert selected by the parsed directory\n";
        return 1;
    }
    std::cout << "PASS: native .strata v1.0 header + ZSTD manifest + expert directory -> "
              << "section-4 descriptor -> ColdStore (" << body_bytes
              << " bytes, absolute offset " << desc.file_off
              << ", dn_off " << dn_off << ")\n";
    return 0;
}
