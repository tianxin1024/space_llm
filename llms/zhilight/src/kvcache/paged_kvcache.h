#pragma once
#include <vector>
#include "core/core.h"
#include "kvcache/kvcache.h"
#include "kvcache/block_allocator.h"
// #include "utils/matrix.hpp"

namespace kvcache {

using namespace bmengine;
// using utils::Matrix2D;

class LogicalBlock {
    std::shared_ptr<BlockTrieNode> phy_block;

public:
    LogicalBlock(std::shared_ptr<BlockTrieNode> block) :
        phy_block(std::move(block)) {
        phy_block->inc_ref_count();
    }

    ~LogicalBlock() {
        phy_block->dec_ref_count();
        if (phy_block->ref_count() == 0)
            phy_block->remove_from_parent();
    }

    BlockTrieNode *get_phy_block() {
        return phy_block.get();
    }
    int32_t phy_block_id() {
        return phy_block->phy_block_id;
    }
    std::vector<int32_t> tokens() {
        return phy_block->tokens();
    }

    bool full() {
        return phy_block->full();
    };
    size_t num_seen_tokens() {
        return phy_block->num_seen_tokens();
    };
    bool can_add_tokens(size_t start, std::vector<int32_t> tokens) {
        return phy_block->can_add_tokens(start, tokens);
    };
    size_t add_tokens(size_t start, std::vector<int32_t> tokens) {
        return phy_block->add_tokens(start, tokens);
    };
};

typedef std::vector<std::vector<std::unique_ptr<LogicalBlock>>> BlockNodeTable;

class PagedKVCache : public KVCache {
    PageConfig page_config;
    std::vector<core::Tensor> key_caches;
    std::vector<core::Tensor> value_caches;
    core::Tensor block_table_; // (batch_size, max_seq_len)
    BlockAllocator block_allocator;
    BlockNodeTable h_block_tables; // (batch_size, logical tables)
    std::vector<size_t> h_seqlens;
    size_t used_logical_blocks = 0;

public:
    PagedKVCache(int num_layers, int num_heads, int dim_head, core::DataType dtype, bool parallel) :
        PagedKVCache(PageConfig{512, 16}, num_layers, num_heads, dim_head, dtype, parallel) {
    }
    PagedKVCache(
        const PageConfig &page_config,
        int num_layers,
        int num_heads,
        int dim_head,
        core::DataType dtype,
        bool parallel);
    ~PagedKVCache();

    const core::Tensor &operator[](int i) const override;
    core::Tensor &operator[](int i) override;

    core::Tensor &key_cache(int i);
    core::Tensor &value_cache(int i);
    const core::Tensor *block_table(int i) const;

    size_t num_sequences() const {
        return batch_size;
    }
    size_t add_sequence(const core::Context &ctx, std::vector<int32_t> prompt_tokens);
    size_t remove_sequence(const core::Context &ctx, size_t seq_id);
    // batched query tokens.
    size_t add_queries(const core::Context &ctx, std::vector<std::vector<int32_t>> query_tokens);

private:
    void sequence_add_tokens(const core::Context &ctx, size_t seq_id, std::vector<int32_t> tokens);
    void resize(const core::Context &ctx, size_t nw_length) override;
    std::unique_ptr<LogicalBlock> clone_block(
        const core::Context &ctx, LogicalBlock *block, size_t n_start);
};

} // namespace kvcache
