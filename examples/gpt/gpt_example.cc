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
#include "models/multi_gpu_gpt/ParallelGpt.h"

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
        ini_name = "../../examples/gpt/gpt_config.ini";
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

    std::string file_name = "../../examples/gpt/start_ids.csv";
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

    AttentionType attention_type = AttentionType::UNFUSED_MHA;

    ParallelGpt<T> gpt = ParallelGpt<T>(
        0, // max_batch_size, QK will adjust the buffer automatically.
        0, // max_seq_len, QK will adjust the buffer automatically.
        0, // max_input_len, QK will adjust the buffer automatically.
        beam_width, head_num, size_per_head, inter_size, decoder_layers,
        0,                                        // expert_num
        0,                                        // moe_k
        {},                                       // moe_layer_index
        vocab_size, start_id, end_id, end_id + 1, // p_prompt_tuning token start id
        PromptLearningType::no_prompt, gptVariantParams{},
        0.0f, // beam_search_diversity_rate,
        0,    // top_k,
        0.0,  // top_p,
        0,    // random_seed,
        1.0f, // temperature,
        0.0f, // len_penalty,
        1.0f, // repetition_penalty,
        stream, &cublas_wrapper, &allocator, false,
        &prop, attention_type, sparse, 0, 0, shared_contexts_ratio);

    int *d_output_ids;
    int *d_sequence_lengths;
    deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);
    std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
        {"output_seq_len",
         Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}},
        {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}},
        {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}},
        {"min_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &min_length}}};

    if (repetition_penalty != 1.0f) {
        input_tensors.insert(
            {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    }
    if (presence_penalty != 0.0f) {
        input_tensors.insert(
            {"presence_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &presence_penalty}});
    }
    if (top_k == 0 && top_p == 0.0f) {
        QK_CHECK(beam_width > 1);
        input_tensors.insert({"beam_search_diversity_rate",
                              Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    } else {
        input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        if (top_p != 0.0f) {
            input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k != 0) {
            input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
        }
    }

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                d_output_ids}},
        {"sequence_length",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}}};

    float *output_log_probs = nullptr;
    float *d_cum_log_probs = nullptr;
    if (is_return_log_probs) {
        deviceMalloc(&output_log_probs, request_batch_size * beam_width * request_output_len);
        output_tensors.insert({"output_log_probs",
                               Tensor{MEMORY_GPU,
                                      TYPE_FP32,
                                      std::vector<size_t>{request_batch_size, beam_width, (size_t)request_output_len},
                                      output_log_probs}});
        deviceMalloc(&d_cum_log_probs, request_batch_size * beam_width);
        output_tensors.insert(
            {"cum_log_probs",
             Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{request_batch_size, beam_width}, d_cum_log_probs}});
        input_tensors.insert({"is_return_context_cum_log_probs",
                              Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_return_context_cum_log_probs}});
    }

    print_mem_usage();
    int ite = 1;
    cudaDeviceSynchronize();

    cudaProfilerStart();
    // warm up
    ite = 1;
    QK_LOG_INFO("warmup time");
    for (int i = 0; i < ite; ++i) {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }
    cudaDeviceSynchronize();

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    ite = 10;
    QK_LOG_INFO("total time");
    for (int i = 0; i < ite; ++i) {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }

    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaProfilerStop();

    printf("[INFO] \n\trequest_batch_size %ld\n\tbeam_width %ld\n\thead_num %ld\n\tsize_per_head %ld\n\ttotal_output_len %d\n "
           "\tdecoder_layers %ld\n\tvocab_size %ld\n\tFT-CPP-decoding-beamsearch-time% .2f ms\n ",
           request_batch_size, beam_width, head_num, size_per_head, total_output_len,
           decoder_layers, vocab_size, ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    if (rank == 0) {
        std::string fName = "out";
        auto outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n",
                   fName.c_str());
        } else {
            size_t outCount = total_output_len * request_batch_size * beam_width;
            int *hBuf = new int[outCount];
            cudaD2Hcpy(hBuf,
                       d_output_ids, outCount);

            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0)) {
                        zeroCount++;
                    }
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (total_output_len) == 0) {
                        outFile << std::endl;
                    }

                    if (i < 10) {
                        printf("%5d ", hBuf[i]);
                    }
                    if ((i + 1) % (total_output_len) == 0 && i < 10) {
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl
                          << "zeroCount = " << zeroCount << std::endl;
            }
            delete[] hBuf;
        }
        outFile.close();

        if (d_cum_log_probs != nullptr) {
            std::string logprob_fname = "logprob.out";
            std::ofstream logprob_file = std::ofstream("logprob.out",
                                                       std::ios::out);
            if (!logprob_file.is_open()) {
                printf("[WARNING] Cannot write results into output file %s \n",
                       logprob_fname.c_str());
            } else {
                size_t cum_log_probs_size = request_batch_size * beam_width;
                printf("[INFO] Writing %ld elements (log probs)\n",
                       cum_log_probs_size);
                float *h_buf = new float[cum_log_probs_size];
                cudaD2Hcpy(h_buf, d_cum_log_probs,
                           cum_log_probs_size);
                for (size_t i = 0; i < cum_log_probs_size;
                     i++) {
                    logprob_file << h_buf[i] << std::endl;
                    if (i < 10) {
                        printf(" %10.6f\n", h_buf[i]);
                    }
                }
                delete[] h_buf;
            }
            logprob_file.close();
        }
    }

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    return;
}
