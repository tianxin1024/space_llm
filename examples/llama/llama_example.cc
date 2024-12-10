#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <cuda_profiler_api.h>

#include "utils/INIReader.h"
#include "utils/cuda_utils.h"

using namespace space_llm;

int main(int argc, char *argv[]) {
    srand(0);
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    printf("finished done!!!\n");
    return 0;
}
