// -*- c++ -*-
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cassert>
#include <fstream>

struct node {
  node *next;
};

template <typename T>
void swap(T &x, T &y) {
  T t = x;
  x = y; y = t;
}

template <typename T>
void shuffle(std::vector<T> &vec, size_t len) {
  for(size_t i = 0; i < len; i++) {
    size_t j = i + (rand() % (len - i));
    swap(vec[i], vec[j]);
  }
}

template <typename T>
size_t partition(T *arr, size_t n) {
  size_t d=0;
  size_t r = rand() % n;
  T p = arr[r];
  arr[r] = arr[n-1];
  arr[n-1] = p;
  
  for(size_t i=0;i<(n-1);i++) {
    if(arr[i] < p) {
      swap(arr[i], arr[d]);
      d++;
    }
  }
  arr[n-1] = arr[d];
  arr[d] = p;
  return d;
}

template <typename T>
void sort(T *arr, size_t len) {
  size_t d;
  if(len <= 16) {
    for(size_t i=1;i<len;i++) {
      size_t j=i;
      while((j > 0) && (arr[j-1] > arr[j])) {
	swap(arr[j-1], arr[j]);
	j--;
      }
    }    
    return;
  }
  d = partition(arr, len);
  sort(arr, d);
  sort(arr+d+1, len-d-1);
}


__global__ void traverse(node **nodes, int64_t *cycles, uint64_t iters) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  node *n = nodes[idx];
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
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;
    n = n->next;    
    iters -= 32;
  }
  cycles[idx] = clock64() - now;
  nodes[idx] = n;
}

int main(int argc, char *argv[]) {
  static const int nthr = 32;
  static const uint64_t max_keys = 1UL<<25;
  static_assert(sizeof(node*)==8);
  std::vector<uint64_t> keys(max_keys);
  node *nodes = nullptr, **nodes_out = nullptr;
  int64_t *cycles = nullptr;
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if(deviceProp.kernelExecTimeoutEnabled) {
    std::cout << "Warning : kernel timeout enabled (long runs will fail)\n";
  }
  double freq = deviceProp.clockRate * 1000.0;
  
  assert(cudaMallocManaged((void**)&nodes, sizeof(node)*max_keys) == cudaSuccess);
  assert(cudaMallocManaged((void**)&nodes_out, sizeof(node*)*nthr) == cudaSuccess);  
  assert(cudaMallocManaged((void**)&cycles, sizeof(int64_t)*nthr) == cudaSuccess);

  std::ofstream out("gpulat.csv");
  
  for(uint64_t n_keys = 1UL<<8; n_keys <= max_keys; n_keys *= 2) {
    for(uint64_t i = 0; i < n_keys; i++) {
      keys[i] = i;
    }
    shuffle(keys, n_keys);
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
    for(int i = 0; i < nthr; i++) {
      nodes_out[i] = h;
    }
    
    if(iters < (1UL<<20)) {
      iters = 1UL<<20;
    }
    
    traverse<<<nthr/32, 32>>>(nodes_out, cycles, iters);
    cudaDeviceSynchronize();
    auto ce = cudaGetLastError();
    if(ce != cudaSuccess) {
      std::cerr << cudaGetErrorString(ce) << "\n";
    }
    sort(cycles, nthr);
    double cpl = static_cast<double>(cycles[nthr/2]) / iters;
    double nspl = (cpl/freq) / (1e-9);
    std::cout << sizeof(node)*n_keys << " bytes, GPU cycles per load "
	      << cpl << ", nanosec per load " << nspl << " \n";
    
    out << (sizeof(node)*n_keys) << ","
	<< cpl << ","<<  nspl << "\n";
    out.flush();
  }
  cudaFree(nodes);
  cudaFree(nodes_out);
  cudaFree(cycles);
  return 0;
}
