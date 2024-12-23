#pragma once
#include <vector>
#include "core/core.h"
#include "kvcache/kvcache.h"
// #include "utils/matrix.hpp"

namespace kvcache {

using namespace bmengine;

struct PageConfig {
    static const size_t DEFAULT_BLOCK_SIZE = 16;

    const size_t num_blocks;
    const size_t page_block_size;
    PageConfig(size_t num_blocks) :
        PageConfig(num_blocks, DEFAULT_BLOCK_SIZE) {
    }
    PageConfig(size_t num_blocks, size_t page_block_size) :
        num_blocks(num_blocks), page_block_size(page_block_size) {
    }
};

struct TokensCmp {
    bool operator()(std::vector<int32_t> lhs, std::vector<int32_t> rhs) const {
        for (auto i = 0; i < std::max(lhs.size(), rhs.size()); i++) {
            // right pad tokens with -1, so shorter segments are always smaller given the common
            // prefix.
            if ((i < lhs.size() ? lhs[i] : -1) > (i < rhs.size() ? rhs[i] : -1)) {
                return true;
            } else if ((i < lhs.size() ? lhs[i] : -1) < (i < rhs.size() ? rhs[i] : -1)) {
                return false;
            } else {
                continue;
            }
        }
        return false;
    }
};

class BlockAllocator;

class BlockTrieNode {
    const PageConfig page_config;
    BlockAllocator *allocator;
    BlockTrieNode *parent;
    std::map<std::vector<int32_t>, std::shared_ptr<BlockTrieNode>, TokensCmp> children;
    std::vector<int32_t> segment;

public:
    const int32_t logical_block_id;
    const int32_t phy_block_id;
    BlockTrieNode(
        const PageConfig &page_config,
        BlockAllocator *allocator,
        int32_t logical_block_id,
        int32_t phy_block_id,
        BlockTrieNode *parent);

    ~BlockTrieNode();

    std::vector<int32_t> tokens() {
        return segment;
    }
    bool full();
    size_t num_seen_tokens();
    bool can_add_tokens(size_t start, std::vector<int32_t> tokens);
    size_t add_tokens(size_t start, std::vector<int32_t> tokens);

    std::shared_ptr<BlockTrieNode> clone(size_t start);

    int32_t ref_count();
    int32_t inc_ref_count();
    int32_t dec_ref_count();
    bool has_child(std::vector<int32_t> tokens) {
        return children.count(tokens) > 0;
    };
    std::shared_ptr<BlockTrieNode> get_child(std::vector<int32_t> tokens) {
        return children.at(tokens);
    };

    std::shared_ptr<BlockTrieNode> reusable_sybling(size_t start, std::vector<int32_t> tokens);
    std::shared_ptr<BlockTrieNode> add_sub_block(std::shared_ptr<BlockTrieNode> child);
    void remove_from_parent();
    BlockTrieNode *reindex_sub_block(
        const std::vector<int32_t> &old_key, const std::vector<int32_t> &new_key);
};

class BlockTrieRoot : public BlockTrieNode {
public:
    BlockTrieRoot(const PageConfig &page_config, BlockAllocator *allocator) :
        BlockTrieNode(page_config, allocator, -1, -1, nullptr) {
    }
    ~BlockTrieRoot() {
    }
};

class BlockAllocator {
public:
    const PageConfig page_config;

    BlockAllocator(const PageConfig &page_config) :
        page_config(page_config),
        block_indices(page_config.num_blocks, 0),
        trie_root_(std::make_unique<BlockTrieRoot>(page_config, this)) {
    }

    std::vector<std::shared_ptr<BlockTrieNode>> find_prefix_blocks(std::vector<int32_t> prefix);
    std::shared_ptr<BlockTrieNode> allocate_block(int32_t logical_block_id, BlockTrieNode *parent);
    BlockTrieNode *trie_root() {
        return trie_root_.get();
    }
    int32_t block_ref(int32_t phy_block_id);
    int32_t inc_block_ref(int32_t phy_block_id);
    int32_t dec_block_ref(int32_t phy_block_id);

private:
    std::vector<int32_t> block_indices; // (phy block_id, ref_counts)
    std::unique_ptr<BlockTrieRoot> trie_root_;
};

} // namespace kvcache
