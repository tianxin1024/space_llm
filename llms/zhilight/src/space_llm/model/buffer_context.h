#pragma once
#include <core/context.h>
#include <core/engine.h>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include "kvcache/kvcache.h"
#include "kvcache/paged_kvcache.h"

namespace model {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Engine;
using bmengine::core::Tensor;
using bmengine::core::WithDevice;
using kvcache::KVCache;
using kvcache::PageConfig;
using kvcache::PagedKVCache;

class ModelBase;

/**
 * Generation buffers managment
 */
class BufferContext {
protected:
    const ModelBase &model_;
    bool parallel_;

public:
    BufferContext(const ModelBase &md, bool parallel = false) :
        model_(md), parallel_(parallel){};
    ~BufferContext() = default;
    BufferContext(BufferContext &&) = default;

    virtual bool is_BSHD() = 0;
    virtual KVCache *buf_k() = 0;
    virtual KVCache *buf_v() = 0;
    virtual Tensor *buf_k(size_t layer) = 0;
    virtual Tensor *buf_v(size_t layer) = 0;

    virtual size_t add_sequence(const Context &ctx, const std::vector<int32_t> &prompt_tokens) = 0;
    virtual size_t remove_sequence(const Context &ctx, size_t seq_id) = 0;
    virtual size_t add_queries(
        const Context &ctx, std::vector<std::vector<int32_t>> query_tokens) = 0;
    virtual const Tensor *block_table(size_t layer) = 0;

    virtual void set_layer_devices(const std::vector<int> &layer_devices) = 0;
    virtual void resize_transformer_buf(const Context &ctx, size_t new_length) = 0;
    virtual size_t kvcache_len() = 0;
};

class TransformerBufferContext : public BufferContext {
private:
    bool BSHD;
    bool cache_paged;

    std::shared_ptr<KVCache> buf_k_;
    std::shared_ptr<KVCache> buf_v_;

    size_t kvcache_len_;

public:
    TransformerBufferContext(
        const ModelBase &md,
        int batch_size = 1,
        bool parallel = false,
        int world_size = 1,
        bool BSHD = false);
    ~TransformerBufferContext();
    TransformerBufferContext(TransformerBufferContext &) = default;

    bool is_BSHD() override {
        return BSHD;
    }
    KVCache *buf_k() override {
        return buf_k_.get();
    }
    KVCache *buf_v() override {
        return buf_v_.get();
    }
    Tensor *buf_k(size_t layer) override;
    Tensor *buf_v(size_t layer) override;

    size_t add_sequence(const Context &ctx, const std::vector<int32_t> &prompt_tokens) override;
    size_t remove_sequence(const Context &ctx, size_t seq_id) override;
    size_t add_queries(const Context &ctx, std::vector<std::vector<int32_t>> query_tokens) override;
    const Tensor *block_table(size_t layer) override;

    void set_layer_devices(const std::vector<int> &layer_devices) override;
    void resize_transformer_buf(const Context &ctx, size_t new_length) override;
    size_t kvcache_len() {
        return kvcache_len_;
    };
};

class PagedBufferContext : public BufferContext {
private:
    std::shared_ptr<PagedKVCache> kvcache_;
    size_t kvcache_len_;

public:
    PagedBufferContext(
        const PageConfig &page_config,
        const ModelBase &md,
        bool parallel = false,
        int world_size = 1);
    ~PagedBufferContext();
    PagedBufferContext(PagedBufferContext &&) = default;

    bool is_BSHD() override {
        return true;
    }
    KVCache *buf_k() override {
        return nullptr;
    }
    KVCache *buf_v() override {
        return nullptr;
    }
    Tensor *buf_k(size_t layer) override;
    Tensor *buf_v(size_t layer) override;

    size_t add_sequence(const Context &ctx, const std::vector<int32_t> &prompt_tokens) override;
    size_t remove_sequence(const Context &ctx, size_t seq_id) override;
    size_t add_queries(const Context &ctx, std::vector<std::vector<int32_t>> query_tokens) override;
    const Tensor *block_table(size_t layer) override;

    void set_layer_devices(const std::vector<int> &layer_devices) override;
    void resize_transformer_buf(const Context &ctx, size_t new_length) override;
    size_t kvcache_len() override;
};
} // namespace model
