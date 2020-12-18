// -*- c++ -*-
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cassert>

struct node {
  node *next;
};

template <typename T>
void swap(T &x, T &y) {
  T t = x;
  x = y; y = t;
}

template <typename T>
void shuffle(std::vector<T> &vec) {
  for(size_t i = 0, len = vec.size(); i < len; i++) {
    size_t j = i + (rand() % (len - i));
    swap(vec[i], vec[j]);
  }
}

__global__ void traverse(node *nn, int64_t *cycles, uint64_t iters) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  volatile node *n = (volatile node*)nn;
  int64_t now = clock64();
  while(iters) {
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;    
    iters -= 8;
  }
  cycles[idx] = clock64() - now;
}

int main(int argc, char *argv[]) {
  static const int nthr = 32;
  static const uint64_t max_keys = 1UL<<24;
  static_assert(sizeof(node*)==8);
  std::vector<uint64_t> keys(max_keys);
  node *nodes = nullptr;
  int64_t *cycles = nullptr;  
  assert(cudaMallocManaged((void**)&nodes, sizeof(node)*max_keys) == cudaSuccess);
  assert(cudaMallocManaged((void**)&cycles, sizeof(int64_t)*nthr) == cudaSuccess);
  
  for(uint64_t n_keys = 1UL<<8; n_keys <= max_keys; n_keys *= 2) {
    for(uint64_t i = 0; i < n_keys; i++) {
      keys[i] = i;
    }
    shuffle(keys);
    node *h = &nodes[keys[0]];
    node *c = h;  
    h->next = h;
    for(uint64_t i = 1; i < n_keys; i++) {
      node *n = &nodes[keys[i]];
      node *t = c->next;
      c->next = n;
      n->next = t;
      c = n;
    }
    uint64_t iters = n_keys*16;
    if(iters > (1UL<<20)) {
      iters = 1UL<<20;
    }
    if(iters < (1UL<<20)) {
      iters = 1UL<<20;
    }
    traverse<<<nthr/32, 32>>>(h, cycles, iters);
    cudaDeviceSynchronize();
    auto ce = cudaGetLastError();
    if(ce != cudaSuccess) {
      std::cerr << cudaGetErrorString(ce) << "\n";
    }
    double t = static_cast<double>(cycles[0]) / iters;
    std::cout << sizeof(node)*n_keys << " bytes, cycles per load " << t << "\n";    
  }
  cudaFree(nodes);  
  cudaFree(cycles);
  return 0;
}
