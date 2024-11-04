#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <cuda_profiler_api.h>

#include "utils/INIReader.h"
#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "layers/attention_layers/BaseAttentionLayer.h"
#include "models/multi_gpu_gpt/ParallelGptWeight.h"

using namespace space_llm;

template <typename T>
void gpt_example(const INIReader reader);

int main(int argc, char *argv[]) {
    srand(0);

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    std::string ini_name;
    if (argc == 2) {
        ini_name = std::string(argv[1]);
    } else {
        ini_name = "../examples/gpt/gpt_config.ini";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("qk_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        gpt_example<float>(reader);
    } else if (data_type == "fp16") {
        gpt_example<half>(reader);
    } else {
        printf("[ERROR] data_type should be fp32, fp16!\n");
        return -1;
    }

    return 0;
}

int read_start_ids(int batch_size, std::vector<int> *v_start_lengths, std::vector<int> *v_start_ids,
                   int &max_input_len, const int end_id, const int beam_width) {
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int> tmp_start_lengths;

    std::string file_name = "../examples/gpt/start_ids.csv";
    std::ifstream start_id_file(file_name, std::ios::in);

    if (start_id_file.is_open()) {
        std::string line;
        int i0 = 0;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string vals;
            int i1 = 0;
            std::vector<int> tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            i0++;
        }
    } else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); ++i) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; ++i) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); ++i) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; ++j) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); ++i) {
        for (int b = 0; b < beam_width; ++b) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); ++j) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
        }
    }

    return 0;
}

template <typename T>
void gpt_example(const INIReader reader) {
    const std::string model_name = reader.Get("qk_instance_hyperparameter", "model_name");
    const size_t max_batch_size = reader.GetInteger("qk_instance_hyperparameter", "max_batch_size");
    const size_t max_seq_len = reader.GetInteger("qk_instance_hyperparameter", "max_seq_len");
    const size_t beam_width = reader.GetInteger("qk_instance_hyperparameter", "beam_width");
    const uint top_k = (uint)reader.GetInteger("qk_instance_hyperparameter", "top_k");
    const float top_p = reader.GetFloat("qk_instance_hyperparameter", "top_p");
    const float temperature = reader.GetFloat("qk_instance_hyperparameter", "temperature");
    const float repetition_penalty = reader.GetFloat("qk_instance_hyperparameter", "repetition_penalty", 1.0f);
    const float presence_penalty = reader.GetFloat("qk_instance_hyperparameter", "presence_penalty", 0.0f);
    const int min_length = reader.GetInteger("qk_instance_hyperparameter", "min_length", 0);
    const std::string model_dir = std::string(reader.Get("qk_instance_hyperparameter", "model_dir"));
    const bool sparse = static_cast<bool>(reader.GetInteger("qk_instance_hyperparameter", "sparse"));
    const float shared_contexts_ratio = reader.GetFloat("qk_instance_hyperparameter", "shared_contexts_ratio", 1.0f);
    const float len_penalty = reader.GetFloat("qk_instance_hyperparameter", "len_penalty");
    const float beam_search_diversity_rate =
        reader.GetFloat("qk_instance_hyperparameter", "beam_search_diversity_rate");
    const unsigned long long int random_seed = 0;

    QK_CHECK_WITH_INFO(
        repetition_penalty == 1.0f || presence_penalty == 0.0f,
        fmtstr("Found ambiguous parameters repetition_penalty (%f) and presence_penalty (%f) "
               "which are mutually exclusive. Please remove one of repetition_penalty or presence_penalty "
               "or set to a default value.",
               repetition_penalty,
               presence_penalty));

    const size_t head_num = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head = reader.GetInteger(model_name, "size_per_head");
    const size_t vocab_size = reader.GetInteger(model_name, "vocab_size");
    const size_t decoder_layers = reader.GetInteger(model_name, "decoder_layers");
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    const int request_output_len = reader.GetInteger("request", "request_output_len");
    // Whether to return the log probabilities of outputs.
    const bool is_return_log_probs = reader.GetBoolean("request", "return_log_probs", false);
    // Whether to include input contexts in computing the cumulative log probabilities.
    const bool is_return_context_cum_log_probs = reader.GetBoolean("request", "context_log_probs", false);
    if (is_return_log_probs && !is_return_context_cum_log_probs) {
        QK_LOG_WARNING("context_log_probs will be ignored since return_log_probs is disabled.");
    }

    const int start_id = 50256;
    const int end_id = 50256;

    const int rank = 0;

    // Read ids of request from file.
    int max_input_len = -1;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    read_start_ids(request_batch_size, &v_start_lengths, &v_start_ids, max_input_len, end_id, 1);

    int *d_input_ids;
    int *d_input_lengths;
    if (max_input_len == 0) {
        // uncoditional case, no input ids, so do nothing.
        d_input_ids = nullptr;
        d_input_lengths = nullptr;
    } else {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }

    const int total_output_len = max_input_len + request_output_len;
    if (total_output_len > (int)max_seq_len) {
        printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len, max_seq_len);
        exit(-1);
    }

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    cublasAlgoMap *cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG);

    Allocator allocator(getDevice());
    std::mutex *cublas_wrapper_mutex = new std::mutex();

    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    } else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    ParallelGptWeight<T> gpt_weights(hidden_units, inter_size, vocab_size, decoder_layers, max_seq_len, 1, 0, 1, 0, 0);

    gpt_weights.loadModel(model_dir);
}
