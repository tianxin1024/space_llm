#include "core/core.h"
#include <numeric>
#include <iostream>
#include <thread>
#include <vector>


int main() {
    bmengine::core::Engine engine(
        {
            {0, 1ll * 1024 * 1024},
            {1, 2ll * 1024 * 1024},
            {2, 3ll * 1024 * 1024},
            {3, 4ll * 1024 * 1024}
        }
    );

    return 0;
}
