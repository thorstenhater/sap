#pragma once

#include "common.hpp"

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

inline constexpr f64* 
set_shm(const char* key, u64 size) {
    auto rc = shm_unlink(key);
    if (rc != 0) perror("UNL");
    
    auto fd = shm_open(key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        perror("SHM");
        return nullptr;
    }
    
    if (ftruncate(fd, size*sizeof(f64)) == -1) {
        perror("TRNC");
        return nullptr;
    }        
    
    auto mem = (char*) mmap((void*)NULL, size*sizeof(double), PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        perror("MMAP");
        return nullptr;
    }

    return (f64*) mem;
} 

inline constexpr void
del_shm(f64* ptr, const char* key, u64 size) {
    auto rc = munmap(ptr, size);
    if (rc < 0) perror("UNMAP");
    auto fd = shm_unlink(key);
    if (fd != 0) perror("UNL");
}
