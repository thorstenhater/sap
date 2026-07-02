#include <iostream>
#include <unistd.h>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>           /* For O_* constants */
#include <thread>

constexpr int N = 32;

using namespace std::chrono_literals;

int main(int argc, char** argv) {
    if (argc != 3) return -1;
    auto fn = argv[1];
    auto md = argv[2][0]; 
    if (md == 'w') {
        auto rc = shm_unlink(fn);
        if (rc != 0) {
            perror("UNL");
            // return -42;
        }

        auto fd = shm_open(fn, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            perror("SHM");
            return -42;
        }

        if (ftruncate(fd, N) == -1) {
            perror("TRNC");
            return -42;
        }        
        
        auto mem = (char*) mmap((void*)NULL, N, PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED) {
            perror("MMAP");
            return -42;
        }

        for (int ix = 0; ix < N; ++ix) mem[ix] = 'a' + ix;
        std::cerr << "Waiting...\n";
        std::this_thread::sleep_for(30s);
    }
    else if (md == 'r') {
    }
}
