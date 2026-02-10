using namespace std;

// #include "PMA.hpp"
//#include <immintrin.h>
#pragma once

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <immintrin.h>
#include <atomic>
#include <unistd.h>

#include <stdint.h>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <string> 

#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "gbbs/bridge.h"

#include "lock.h"

#include <immintrin.h>
#include <iostream>
#include <iomanip>    

#define temp_pfor for 
namespace graphstore {

#if defined(LONG)
typedef uint64_t uint_t;
typedef int64_t int_t;
#else
typedef uint32_t uint_t;
typedef int32_t int_t;
#endif

#define NULL_VAL (UINT32_MAX)
#define SENT_VAL (UINT32_MAX -1)

typedef struct _edge {
  uint32_t dest;
  uint32_t value;
} edge_t;

typedef struct _pair_double {
  double x;
  double y;
} pair_double;

typedef struct _pair_loc {
  uint_t loc;
  uint32_t dest;
  uint32_t value;
} pair_loc;

typedef struct _node {
  // beginning and end of the associated region in the edge list
  uint_t beginning;     // deleted = max int
  uint_t end;           // end pointer is exclusive
  uint32_t num_neighbors; // number of edgess with this node as source
} node_t;

typedef struct edge_list {
  uint_t N;
  uint_t non_null;
  uint32_t H;
  uint32_t logN;
  uint32_t loglogN;
  uint32_t mask_for_leaf;
  uint32_t leaf_num;
  uint32_t * vals;
  uint32_t * dests;
  std::vector<std::vector<pair_loc>> temp_edges;
  LOCK *leaf_locks;
  bool *dist_flag;

  double density_limit;
} edge_list_t;

class PMA {
public:
  // data members
  edge_list_t edges;
  std::vector<node_t> nodes;
  uint_t full_num;

  double upper_density_bound[32];
  double lower_density_bound[32];

  PMA(uint32_t init_n = 16);
  PMA(PMA &other);
  ~PMA();
  void double_list();
  void half_list();

  void slide_right(uint_t index, uint32_t *vals, uint32_t *dests);
  void slide_left(uint_t index, uint32_t *vals, uint32_t *dests);
  void redistribute(uint_t index, uint64_t len);
  void redistribute_par(uint_t index, uint64_t len);
  void redistribute_with_temp(uint_t index, uint64_t len);
  uint_t fix_sentinel(uint32_t node_index, uint_t in);
  void print_array(uint_t index, uint32_t len);
  uint32_t find_value(uint32_t src, uint32_t dest);
  void print_graph();
  void add_node();

  void build_from_edges(uint32_t *srcs, uint32_t *dests, uint8_t * pma_edges, uint64_t vertex_count, uint64_t edge_count, uint32_t* additional_degrees);

  // merge functions in original PMA with no val
  uint32_t binary_find_leaf(uint32_t begin_id, uint32_t end_id, uint32_t dest);
  uint_t binary_find_elem(uint_t begin_idx, uint_t end_idx, uint32_t dest);

  void add_edge_batch_wrapper(pair_uint *es, uint64_t edge_count);
  bool add_edge_once(uint32_t src, uint32_t dest, uint32_t value);
  int add_edge_lock(uint32_t src, uint32_t dest, uint32_t value,  uint32_t leaf_begin, uint32_t leaf_end, uint32_t leaf_id);
  bool add_edge_serial(uint32_t src, uint32_t dest, uint32_t value);

  void remove_edge_batch_wrapper(pair_uint *es, uint64_t edge_count);
  void remove_edge_fast(uint32_t src, uint32_t dest);
  bool remove_edge_once(uint32_t src, uint32_t loc_to_remove);
  int remove_edge_lock(uint32_t src, uint32_t dest, uint32_t value, uint32_t leaf_begin, uint32_t leaf_end, uint32_t leaf_id);
  bool remove_edge_serial(uint32_t src, uint32_t dest, uint32_t value);

  void insert(uint_t index, uint32_t elem_dest, uint32_t elem_value, uint32_t src);
  bool insert_dist(uint_t leaf_index);
  bool insert_dist_serial(uint_t leaf_index);
  bool insert_dist_batch(uint32_t leaf_id);

  void remove(uint_t index, uint32_t elem_dest, uint32_t src);
  bool remove_dist_lock(uint_t leaf_index);
  // bool remove_dist_batch();
  bool remove_dist_batch(uint_t leaf_index);
  bool remove_dist_serial(uint_t leaf_index);
  
  void grab_locks_range(uint_t begin, uint_t len);
  void release_locks_range(uint_t begin, uint_t len);

  uint64_t get_size();
  uint64_t get_n();
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> get_edges();
  void clear();
  uint32_t find_contaning_node(uint_t index);

  uint32_t num_neighbors(uint32_t node) {
    return nodes[node].num_neighbors;
  }

  uint64_t num_edges() {
    uint64_t num = 0;
    for (uint64_t i = 0; i < get_n(); i++) {
      num += num_neighbors(i);
    }
    return num;
  }
};

// same as find_leaf, but does it for any level in the tree
// index: index in array
// len: length of sub-level.
static inline uint_t find_node(uint_t index, uint_t len) { return (index / len) * len; }

static uint_t find_leaf(edge_list_t *list, uint_t index) {
  return index & list->mask_for_leaf;
}


static uint_t find_prev_valid(uint32_t volatile  * volatile dests, uint_t start) {
  while (dests[start] == NULL_VAL) {
    start--;
  }
  return start;
}

uint_t inline next_leaf(uint_t index, int loglogN) {
  return ((index >> loglogN) + 1) << (loglogN);
}

void inline PMA::clear() {
  printf("clear called\n");
  uint_t n = 0;
  
  free((void*)edges.vals);
  free((void*)edges.dests);
  free((void*)edges.leaf_locks);
  free((void*)edges.dist_flag);
  edges.N = 2UL << bsr_word(n);
  // printf("%d\n", bsf_word(list->N));
  edges.loglogN = bsr_word(bsr_word(edges.N) + 1);
  edges.logN = (1 << edges.loglogN);
  edges.leaf_num = edges.N / edges.logN;
  edges.H = bsr_word(edges.leaf_num);
}

vector<tuple<uint32_t, uint32_t, uint32_t>> inline PMA::get_edges() {
  uint64_t n = get_n();
  vector<tuple<uint32_t, uint32_t, uint32_t>> output;

  for (uint_t i = 0; i < n; i++) {
    uint_t start = nodes[i].beginning;
    uint_t end = nodes[i].end;
    for (uint_t j = start + 1; j < end; j++) {
      if (edges.dests[j]!=NULL_VAL) {
        output.push_back(
            make_tuple(i, edges.dests[j], edges.vals[j]));
      }
    }
  }
  return output;
}

uint64_t inline PMA::get_n() {
  uint64_t size = nodes.size();
  return size;
}

uint64_t inline PMA::get_size() {
  uint64_t size = nodes.capacity() * sizeof(node_t);
  // printf("node size = %lu\n", size);
  size += sizeof(PMA);
  // printf("PMA size = %lu\n", sizeof(PMA));
  size += (uint64_t)edges.N * sizeof(edge_t);
  printf("edges.N = %lu\n", (uint64_t)edges.N);
  // printf("edge size = %lu\n", (uint64_t)edges.N * sizeof(edge_t));
  size += sizeof(LOCK) * edges.leaf_num;
  // printf("lock size = %lu\n", sizeof(LOCK) * edges.leaf_num);
  size += sizeof(bool) * edges.leaf_num;
  // printf("dist flag size = %lu\n", sizeof(bool) * edges.leaf_num);
  // printf("lock size = %lu\n", sizeof(LOCK));
  size += edges.temp_edges.size() * sizeof(std::vector<pair_loc>);
  printf("temp edge size = %lu\n", edges.temp_edges.size() * sizeof(std::vector<pair_loc>));
  return size;
}

/*
void inline print_array(edge_list_t *edges) {
  printf("N = %d, logN = %d\n", edges->N, edges->logN);
  for (uint_t i = 0; i < edges->N; i++) {
    if (edges->dests[i]==NULL_VAL) {
      printf("%d-x ", i);
    } else if ((edges->dests[i]==SENT_VAL) || i == 0) {
      uint32_t value = edges->vals[i];
      if (value == NULL_VAL) {
        value = 0;
      }
      printf("\n%d-s(%u):(?, ?) ", i, value);
    } else {
      printf("%d-(%d, %u) ", i, edges->dests[i], edges->vals[i]);
    }
  }
  printf("\n\n");
} 

void inline PMA::print_array(uint64_t worker_num) {
  for (uint_t i = 0; i < edges.N; i++) {
    if (edges.dests[i]==NULL_VAL) {
      printf("%d-x ", i);
    } else if ((edges.dests[i] == SENT_VAL) || i == 0) {
      uint32_t value = edges.vals[i];
      if (value == NULL_VAL) {
        value = 0;
      }
    } else {
      printf("%d-(%d, %u) ", i, edges.dests[i], edges.vals[i]);
    }
  }
  printf("\n\n");
}

*/

inline void PMA::print_array(uint_t index, uint32_t len) {
  for (uint_t i = index; i < index + len; i++) {
#if defined(LONG)
    printf("(%lu, %u) ", i, edges.dests[i]);
#else
    printf("(%u, %u) ", i, edges.dests[i]);
#endif
  }
  printf("\n");
}

uint_t inline get_density_count(edge_list_t *list, uint_t index, uint_t len) {
  // fater without paralleliszation since it gets properly vectorized
  uint32_t * dests = (uint32_t *) list->dests;
  uint_t full = 0;
#ifdef  __AVX__
  if (len >= 8) {
    uint_t null_count = 0;
    for (uint_t i = index; i < index+len; i+=8) {
      //TODO if we keep things aligned this could be faster, but then we cant use realloc in double
      __m256i a = _mm256_loadu_si256((__m256i *)& dests[i]);
      __m256i b =  _mm256_set1_epi32(NULL_VAL);
      uint32_t add = __builtin_popcount(_mm256_movemask_ps((__m256)_mm256_cmpeq_epi32(a, b)));
      null_count += add;
    }
    full = len - null_count;
    return full;
  }
#endif
  for (uint_t i = index; i < index+len; i+=4) {
      uint32_t add = (dests[i]!=NULL_VAL) + (dests[i+1]!=NULL_VAL) + (dests[i+2]!=NULL_VAL) + (dests[i+3]!=NULL_VAL);
      //__sync_fetch_and_add(&full, add);
      full+=add;
  }
  return full;
}

// get density of a node
double inline get_density(edge_list_t *list, uint_t index, uint_t len) {
  double full_d = (double)get_density_count(list, index, len);
  return full_d / len;
}

double inline get_density_with_temp(edge_list_t *list, uint_t index, uint_t len) {
    uint_t main_count = get_density_count(list, index, len);
    uint_t temp_count = 0;
    uint_t leaf_start = index / list->logN;
    uint_t leaf_end = (index + len - 1) / list->logN;
    for (uint_t leaf_id = leaf_start; leaf_id <= leaf_end; ++leaf_id) {
        temp_count += list->temp_edges[leaf_id].size();
    }

    return double(main_count + temp_count) / double(len);
}

bool inline check_no_full_leaves(edge_list_t *list, uint_t index, uint_t len) {
  for (uint_t i = index; i < index + len; i+= list->logN) {
    bool full = true;
    for (uint_t j = i; j < i + list->logN; j++) {
       if (list->dests[j]==NULL_VAL) {
        full = false;
      }
    }
    if (full) {
      return false;
    }
  }
  return true;
}

// height of this node in the tree
int inline get_depth(edge_list_t *list, uint_t len) { return bsr_word(list->N / len); }

// when adjusting the list size, make sure you're still in the
// density bound
pair_double inline density_bound(edge_list_t *list, int depth) {
  pair_double pair;

  // between 1/4 and 1/2
  // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
  // between 1/8 and 1/4
  pair.x = 1.0 / 4.0 - ((.125 * depth) / list->H);
  pair.y = 3.0 / 4.0 + ((.25 * depth) / list->H);
  if (pair.y > list->density_limit) {
    pair.y = list->density_limit-.001;
  }
  return pair;
}

//TODO make it so the first element is known to always be the first element and don't special case it so much
//returns where me have to start checking for sentinals again
// ths is just the start + the degree so we can run some fasts paths
uint_t inline PMA::fix_sentinel(uint32_t node_index, uint_t in) {
  // we know the first sentinal will never move so we just ignore it
  // assert(node_index > 0);
  nodes[node_index - 1].end = in;

  nodes[node_index].beginning = in;
  if (node_index == nodes.size() - 1) {
    nodes[node_index].end = edges.N - 1;
  }
  return nodes[node_index].beginning + nodes[node_index].num_neighbors;
}


// Evenly redistribute elements in the ofm, given a range to look into
// index: starting position in ofm structure
// len: area to redistribute
void inline PMA::redistribute(uint_t index, uint64_t len) {
  uint32_t *space_vals;
  uint32_t *space_dests;
  uint32_t * vals = (uint32_t *) edges.vals;
  uint32_t  * dests = (uint32_t *) edges.dests;
  uint_t j = 0;
  space_vals = (uint32_t *)malloc(len * sizeof(*(edges.vals)));
  space_dests = (uint32_t *)malloc(len * sizeof(*(edges.dests)));

  //TODO could parralize if get_density_count gave us more data, but doesn't seem to be a bottle neck
  // could get better cache behavior if I go back and forth with reading and writing
  // assert(len >= 8);
  // AVX code seems to have a bug, but I can't find it s leaving it in the iffalse
  for (uint_t i = index; i < index + len; i+=8) {
    for (uint_t k = i; k < i+8; k++) {
      space_vals[j] = vals[k];
      space_dests[j] = dests[k];
      // counting non-null edges
      j += (space_dests[j]!=NULL_VAL);
    }
    memset (__builtin_assume_aligned((void*)&vals[i], 32), 0, 8*sizeof(uint32_t));
    //setting by byte, but NULL_VAL is all ones so it is fine
    memset (__builtin_assume_aligned((void*)&dests[i], 32), NULL_VAL, 8*sizeof(uint32_t));
  }
  
  // // assert( ((double)j)/len <= edges.density_limit);
  // if (((double)j)/len > edges.density_limit) {
    // printf("len = %lu, j = %lu, density = %f, limit = %f\n", len, j, ((double)j)/len, ((double)(edges.logN-1)/edges.logN));
    // print_array(index, len);
  // }

  uint_t num_leaves = len >> edges.loglogN;
  uint_t count_per_leaf = j / num_leaves;
  uint_t extra = j % num_leaves;
  __builtin_prefetch ((void *)&nodes, 0, 3);

  // parallizing does not make it faster
  uint_t end_sentinel = 0;
  for (uint_t i = 0; i < num_leaves; i++) {
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    uint_t in = index + (edges.logN * (i));
    uint_t j2 = count_per_leaf*i + min(i,extra);
    //TODO could be parallized, but normally only up to size 32
    uint_t j3 = j2;
    memcpy(__builtin_assume_aligned((void*)&vals[in],16), (void*)&space_vals[j2], count_for_leaf*sizeof(uint32_t));
    if (end_sentinel > in + count_for_leaf) {
      memcpy(__builtin_assume_aligned((void*)&dests[in], 16), (void*)&space_dests[j2], count_for_leaf*sizeof(uint32_t));
    } else { 
      for (uint_t k = in; k < count_for_leaf+in; k++) {
        dests[k] = space_dests[j3];
        if (dests[k]==SENT_VAL) {
          // fixing pointer of node that goes to this sentinel
          uint32_t node_index = vals[k];
          end_sentinel = fix_sentinel(node_index, k);
        }
        j3++;
      }
    }
  }
  free(space_dests);
  free(space_vals);
  // // assert(check_no_full_leaves(&edges, index, len));
}

void inline PMA::redistribute_with_temp(uint_t index, uint64_t len) {
  // printf("redistribute_with_temp called on index %lu, len %lu\n", index, len);
  uint32_t *space_vals = (uint32_t *)malloc(len * sizeof(*(edges.vals)));
  uint32_t *space_dests = (uint32_t *)malloc(len * sizeof(*(edges.dests)));
  uint32_t *vals = (uint32_t *) edges.vals;
  uint32_t  *dests = (uint32_t *) edges.dests;
  uint_t j = 0;

  for (uint_t i = index; i < index + len; i += edges.logN) {
    uint32_t leaf_id = i / edges.logN;
    if (!edges.temp_edges[leaf_id].empty()) {
      // printf("Merging temp edges into leaf %u with %lu temp edges\n", leaf_id, edges.temp_edges[leaf_id].size());
      uint_t size = edges.temp_edges[leaf_id].size() + edges.logN;
      pair_loc *merge = new pair_loc[size];
      for (uint_t k = 0; k < edges.logN; k++) {
        merge[k] = {i + k, dests[i + k], vals[i + k]};
      }
      for (uint_t k = 0; k < edges.temp_edges[leaf_id].size(); k++) {
        merge[k + edges.logN] = edges.temp_edges[leaf_id][k];
      }
      std::sort(merge, merge + size,
          [](const pair_loc& a, const pair_loc& b) {
          if (a.loc != b.loc) return a.loc < b.loc;
            return a.dest < b.dest;
      });

      for (uint_t k = 0; k < size; k++) {
        space_vals[j] = merge[k].value;
        space_dests[j] = merge[k].dest;
        // counting non-null edges
        j += (space_dests[j]!=NULL_VAL);
      }
      free((void*)merge);
      edges.temp_edges[leaf_id].clear();
    } else {
      for (uint_t k = i; k < i + edges.logN; k++) {
        space_vals[j] = vals[k];
        space_dests[j] = dests[k];
        // counting non-null edges
        j += (space_dests[j]!=NULL_VAL);
      }
    }
    memset (__builtin_assume_aligned((void*)&vals[i], 32), 0, edges.logN*sizeof(uint32_t));
    //setting by byte, but NULL_VAL is all ones so it is fine
    memset (__builtin_assume_aligned((void*)&dests[i], 32), NULL_VAL, edges.logN*sizeof(uint32_t));
  }
  
  // printf("Redistributing %lu elements\n", j);
  uint_t num_leaves = len >> edges.loglogN;
  uint_t count_per_leaf = j / num_leaves;
  uint_t extra = j % num_leaves;
  __builtin_prefetch ((void *)&nodes, 0, 3);

  // parallizing does not make it faster
  uint_t end_sentinel = 0;
  for (uint_t i = 0; i < num_leaves; i++) {
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    uint_t in = index + (edges.logN * (i));
    uint_t j2 = count_per_leaf*i + min(i,extra);
    //TODO could be parallized, but normally only up to size 32
    uint_t j3 = j2;
    memcpy(__builtin_assume_aligned((void*)&vals[in], 32), (void*)&space_vals[j2], count_for_leaf*sizeof(uint32_t));
    if (end_sentinel > in + count_for_leaf) {
      memcpy(__builtin_assume_aligned((void*)&dests[in], 32), (void*)&space_dests[j2], count_for_leaf*sizeof(uint32_t));
    } else { 
      for (uint_t k = in; k < count_for_leaf+in; k++) {
        dests[k] = space_dests[j3];
        if (dests[k]==SENT_VAL) {
          // fixing pointer of node that goes to this sentinel
          uint32_t node_index = vals[k];
          end_sentinel = fix_sentinel(node_index, k);
        }
        j3++;
      }
    }
  }
  free(space_dests);
  free(space_vals);
  // // assert(check_no_full_leaves(&edges, index, len));
}

void inline PMA::redistribute_par(uint_t index, uint64_t len) {
  // printf("redistribute called on index %lu, len %lu\n", index, len);
  // assert(find_leaf(&edges, index) == index);
  
  uint32_t *space_vals;
  uint32_t *space_dests;
  uint32_t * vals = (uint32_t *) edges.vals;
  uint32_t  * dests = (uint32_t *) edges.dests;
  uint_t j = 0;
  
  space_vals = (uint32_t *)malloc(len * sizeof(*(edges.vals)));
  space_dests = (uint32_t *)malloc(len * sizeof(*(edges.dests)));


  //TODO could parralize if get_density_count gave us more data, but doesn't seem to be a bottle neck
  // could get better cache behavior if I go back and forth with reading and writing
  // assert(len >= 8);
  // AVX code seems to have a bug, but I can't find it s leaving it in the iffalse
  for (uint_t i = index; i < index + len; i+=8) {
    for (uint_t k = i; k < i+8; k++) {
      space_vals[j] = vals[k];
      space_dests[j] = dests[k];
      // counting non-null edges
      j += (space_dests[j]!=NULL_VAL);
    }
    memset (__builtin_assume_aligned((void*)&vals[i], 32), 0, 8*sizeof(uint32_t));
    //setting by byte, but NULL_VAL is all ones so it is fine
    memset (__builtin_assume_aligned((void*)&dests[i], 32), NULL_VAL, 8*sizeof(uint32_t));
  }

  uint_t num_leaves = len >> edges.loglogN;
  uint_t count_per_leaf = j / num_leaves;
  uint_t extra = j % num_leaves;
  __builtin_prefetch ((void *)&nodes, 0, 3);

  // parallizing does not make it faster
  uint_t end_sentinel = 0;
  parlay::parallel_for (0, num_leaves, [&](uint64_t i) {
  // for (uint_t i = 0; i < num_leaves; i++) {
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    uint_t in = index + (edges.logN * (i));
    uint_t j2 = count_per_leaf*i + min(i,extra);
    //TODO could be parallized, but normally only up to size 32
    uint_t j3 = j2;
    memcpy(__builtin_assume_aligned((void*)&vals[in],16), (void*)&space_vals[j2], count_for_leaf*sizeof(uint32_t));
    if (end_sentinel > in + count_for_leaf) {
      memcpy(__builtin_assume_aligned((void*)&dests[in], 16), (void*)&space_dests[j2], count_for_leaf*sizeof(uint32_t));
    } else { 
      for (uint_t k = in; k < count_for_leaf+in; k++) {
        dests[k] = space_dests[j3];
        if (dests[k]==SENT_VAL) {
          // fixing pointer of node that goes to this sentinel
          uint32_t node_index = vals[k];
          end_sentinel = fix_sentinel(node_index, k);
        }
        j3++;
      }
    }
  // }
  });
  
  free(space_dests);
  free(space_vals);
  // // assert(check_no_full_leaves(&edges, index, len));
}

//TODO pass in subcounts and do redistibute_par when big
void PMA::double_list() {
  printf("Double list \n");
  edges.N = edges.N * 2;
  edges.loglogN = bsr_word(bsr_word(edges.N) + 1);
  edges.logN = (1 << edges.loglogN);
  edges.mask_for_leaf = ~(edges.logN - 1);
  // assert(edges.logN > 0);
  edges.density_limit = ((double) edges.logN - 1)/edges.logN;
  edges.leaf_num = edges.N / edges.logN;
  edges.H = bsr_word(edges.leaf_num);
  for (uint32_t i = 0; i <= edges.H; i++) {
    upper_density_bound[i] = density_bound(&edges, i).y;
    lower_density_bound[i] = density_bound(&edges, i).x;
  }
  
  uint32_t * vals = (uint32_t *)edges.vals;
  uint32_t * dests = (uint32_t *)edges.dests;
  uint32_t *space_vals = (uint32_t *)malloc(edges.N / 2 * sizeof(*(edges.vals)));
  uint32_t *space_dests = (uint32_t *)malloc(edges.N / 2 * sizeof(*(edges.dests)));
  
  uint_t j = 0;
  for (uint_t i = 0; i < edges.N / 2; i+=8) {
    for (uint_t k = i; k < i+8; k++) {
      space_vals[j] = vals[k];
      space_dests[j] = dests[k];
      // counting non-null edges
      j += (space_dests[j]!=NULL_VAL);
    }
  }
  
  uint_t num_leaves = edges.leaf_num;
  uint_t count_per_leaf = j / num_leaves;
  uint_t extra = j % num_leaves;
  
  uint32_t *new_dests = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.dests)));
  uint32_t *new_vals = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.vals)));
  LOCK *new_leaf_locks = (LOCK *)aligned_alloc(32, num_leaves * sizeof(*(edges.leaf_locks)));
  bool *new_dist_flag = (bool *)aligned_alloc(32, num_leaves * sizeof(*(edges.dist_flag)));
  edges.temp_edges.resize(num_leaves);  

  parlay::parallel_for (0, edges.N, [&](uint64_t i) {
    new_vals[i] = 0; // setting to null
    new_dests[i] = NULL_VAL; // setting to null
  });

  parlay::parallel_for(0, num_leaves, [&](uint64_t i) {
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    uint_t in = ((i) << edges.loglogN);
    uint_t j2 = count_per_leaf*i +min(i,extra);
    uint_t j3 = j2;
    for(uint_t k = in; k < count_for_leaf+in; k++) {
      // // assert(j2 < num_elts);
      new_vals[k] = space_vals[j2];
      j2++;
    }
    for (uint_t k = in; k < count_for_leaf+in; k++) {
      new_dests[k] = space_dests[j3];
      if (new_dests[k]==SENT_VAL) {
        uint32_t node_index = space_vals[j3];
        fix_sentinel(node_index, k);
      }
      j3++;
    }
    new_leaf_locks[i].init();
    new_dist_flag[i] = false;
  });

  free(space_dests);
  free(space_vals);

  free((void*)edges.vals);
  edges.vals = new_vals;
  free((void*)edges.dests);
  edges.dests = new_dests;
  free((void*)edges.leaf_locks);
  edges.leaf_locks = new_leaf_locks;
  free((void*)edges.dist_flag);
  edges.dist_flag = new_dist_flag;
}

void inline PMA::half_list() {
  printf("Half list \n");
  edges.N = edges.N / 2;
  edges.loglogN = bsr_word(bsr_word(edges.N) - 1);
  edges.logN = (1 << edges.loglogN);
  edges.mask_for_leaf = ~(edges.logN - 1);
  // assert(edges.logN > 0);
  edges.density_limit = ((double) edges.logN - 1)/edges.logN;
  edges.leaf_num = edges.N / edges.logN;
  edges.H = bsr_word(edges.leaf_num);
  for (uint32_t i = 0; i <= edges.H; i++) {
    upper_density_bound[i] = density_bound(&edges, i).y;
    lower_density_bound[i] = density_bound(&edges, i).x;
  }

  uint32_t * vals = (uint32_t *)edges.vals;
  uint32_t * dests = (uint32_t *)edges.dests;
  uint32_t *space_vals = (uint32_t *)malloc(edges.N * sizeof(*(edges.vals)));
  uint32_t *space_dests = (uint32_t *)malloc(edges.N * sizeof(*(edges.dests)));

  uint_t j = 0;
  for (uint_t i = 0; i < edges.N * 2; i+=8) {
    for (uint_t k = i; k < i+8; k++) {
      space_vals[j] = vals[k];
      space_dests[j] = dests[k];
      // counting non-null edges
      j += (space_dests[j]!=NULL_VAL);
    }
  }

  uint_t num_leaves = edges.leaf_num;
  uint_t count_per_leaf = j / num_leaves;
  uint_t extra = j % num_leaves;
  
  uint32_t *new_dests = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.dests)));
  uint32_t *new_vals = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.vals)));
  LOCK *new_leaf_locks = (LOCK *)aligned_alloc(32, num_leaves * sizeof(*(edges.leaf_locks)));
  bool *new_dist_flag = (bool *)aligned_alloc(32, num_leaves * sizeof(*(edges.dist_flag)));
  edges.temp_edges.resize(num_leaves);  

  parlay::parallel_for (0, edges.N, [&](uint64_t i) {
    new_vals[i] = 0; // setting to null
    new_dests[i] = NULL_VAL; // setting to null
  });

  parlay::parallel_for(0, num_leaves, [&](uint64_t i) {
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    uint_t in = ((i) << edges.loglogN);
    uint_t j2 = count_per_leaf*i +min(i,extra);
    uint_t j3 = j2;
    for(uint_t k = in; k < count_for_leaf+in; k++) {
      // assert(j2 < num_elts);
      new_vals[k] = space_vals[j2];
      j2++;
    }
    for (uint_t k = in; k < count_for_leaf+in; k++) {
      new_dests[k] = space_dests[j3];
      if (new_dests[k]==SENT_VAL) {
        uint32_t node_index = space_vals[j3];
        fix_sentinel(node_index, k);
      }
      j3++;
    }
    new_leaf_locks[i].init();
    new_dist_flag[i] = false;
  });

  free(space_dests);
  free(space_vals);

  free((void*)edges.vals);
  edges.vals = new_vals;
  free((void*)edges.dests);
  edges.dests = new_dests;
  free((void*)edges.leaf_locks);
  edges.leaf_locks = new_leaf_locks;
  free((void*)edges.dist_flag);
  edges.dist_flag = new_dist_flag;
}


// index is the beginning of the sequence that you want to slide right.
void inline PMA::slide_right(uint_t index, uint32_t *vals, uint32_t *dests) {
  uint32_t next_leaf = find_leaf(&edges, index) + edges.logN;
  // assert(next_leaf <= edges.N);
  uint32_t el_val = vals[index];
  uint32_t el_dest = dests[index];
  dests[index] = NULL_VAL;
  vals[index] = 0;

  index++;

  while (index < next_leaf && (dests[index]!=NULL_VAL)) {
    uint32_t temp_val = vals[index];
    uint32_t temp_dest = dests[index];
    vals[index] = el_val;
    dests[index] = el_dest;
    if (el_dest == SENT_VAL) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = el_val;
      fix_sentinel(node_index, index);
    }
    el_val = temp_val;
    el_dest = temp_dest;
    index++;
  }

  if (el_dest == SENT_VAL) {
    // fixing pointer of node that goes to this sentinel
    uint32_t node_index = el_val;
    fix_sentinel(node_index, index);
  }

  // TODO There might be an issue with this going of the end sometimes
  // assert(index != edges.N);

  vals[index] = el_val;
  dests[index] = el_dest;
}


// index is the beginning of the sequence that you want to slide left.
// the element we start at will be deleted
void inline PMA::slide_left(uint_t index, uint32_t *vals, uint32_t *dests) {
  uint32_t next_leaf = find_leaf(&edges, index) + edges.logN;
  // assert(next_leaf <= edges.N);
  while (index + 1 < next_leaf) {
    uint32_t temp_val = vals[index+1];
    uint32_t temp_dest = dests[index+1];
    vals[index] = temp_val;
    dests[index] = temp_dest;
    if (temp_dest == SENT_VAL) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = temp_val;
      fix_sentinel(node_index, index);
    }
    if (dests[index] == NULL_VAL) {
      break;
    }
    index++;
  }

  if (index + 1 == next_leaf) {
    vals[index] = 0;
    dests[index] = NULL_VAL;
  }

  // assert(index != edges.N);
}

// important: make sure start, end don't include sentinels
// returns the index of the smallest element bigger than you in the range
// [start, end)
// if no such element is found, returns end (because insert shifts everything to
// the right)
uint_t inline binary_search(const edge_list_t *list, uint32_t elem_dest, uint32_t elem_val, uint_t start,
                       uint_t end) {
  // assert(start <= end);
  uint32_t *dests = (uint32_t *) list->dests;

  uint_t mid = (start + end) / 2;
  while (start + 1 < end) {
    
    __builtin_prefetch ((void *)&dests[(mid+end)/2], 0, 3);
    __builtin_prefetch ((void *)&dests[(start + mid)/2], 0, 3);
    // printf("start = %d, end = %d, dest = %d, mid = %d, val =%u\n", start, end, elem_dest, mid, elem_val);
    uint32_t item_dest = dests[mid];

    //if is_null
    if (item_dest==NULL_VAL) {
      // first check the next leaf
      uint_t check = next_leaf(mid, list->loglogN);
      //TODO deal with check is null
      if (check > end) {
        end = mid;
        mid = (start + end) / 2;
        __builtin_prefetch ((void *)&dests[mid], 0, 3);
        continue;
      }
      // if is_null
      if (dests[check]==NULL_VAL) {
        uint_t early_check = find_prev_valid(dests, mid);
        // if we found the sentinel, go right after it
        if (dests[early_check] == SENT_VAL) {
          return early_check + 1;
        }
        if (early_check < start) {
          start = mid;
          mid = (start + end) / 2;
          __builtin_prefetch ((void *)&dests[mid], 0, 3);
          continue;
        }  
        check = early_check;
      } 
      // printf("check = %d\n", check);
      uint32_t dest = dests[check];
      if (elem_dest == dest) {
        // cleanup before return
        return check;
      } else if (elem_dest < dest) {
        end = find_prev_valid(dests, mid) + 1;

      } else {
        if (check == start) {
          start = check + 1;
        } else {
          start = check;
        }
        // otherwise, searched for item is more than current and we set start
      }
      mid = (start + end) / 2;
      __builtin_prefetch ((void *)&dests[mid], 0, 3);
      continue;
    }

    if (elem_dest < item_dest) {
      end = mid; // if the searched for item is less than current item, set end
      mid = (start + end) / 2;
    } else if (elem_dest > item_dest) {
      start = mid;
      mid = (start + end) / 2;
      // otherwise, sesarched for item is more than current and we set start
    } else if (elem_dest == item_dest) {  // if we found it, return
      // cleanup before return
      return mid;
    }
  }
  if (end < start) {
    start = end;
  }
  // assert(start >= 0);

  //trying to encourage the packed left property so if they are both null go to the left
  if ((dests[start]==NULL_VAL) && (dests[end]==NULL_VAL)) {
    end = start;
  }

  // handling the case where there is one element left
  // if you are leq, return start (index where elt is)
  // otherwise, return end (no element greater than you in the range)
  // printf("start = %d, end = %d, n = %d\n", start,end, list->N);
  if (elem_dest <= dests[start] && (dests[start]!=NULL_VAL)) {
    end = start;
  }
  // cleanup before return

  return end;
}

inline uint_t sequential_search(const edge_list_t *list, uint32_t dest, uint_t start, uint_t end) {
  uint32_t *dests = (uint32_t *) list->dests;
  for (uint32_t i = start; i < end; ++i) {
    if (dests[i] >= dest) {
      return i;
    }
  }
}

uint32_t inline PMA::find_value(uint32_t src, uint32_t dest) {

  uint32_t e_value = 0;
  uint32_t e_dest = dest;

  uint_t loc =
      binary_search(&edges, e_dest, e_value, nodes[src].beginning + 1, nodes[src].end);
  //printf("loc = %d, looking for %u, %u, found %u, %u\n",loc, src, dest, edges.dests[loc], edges.vals[loc]);
  e_dest = edges.dests[loc];
  e_value = edges.vals[loc];

  //TODO probably don't need the first check since we will never look for null
  if ((e_dest != NULL_VAL) && e_dest == dest) {
    return e_value;
  } else {
    return 0;
  }
}

uint32_t inline PMA::find_contaning_node(uint_t index) {
  uint32_t start = 0; 
  uint32_t end = nodes.size()-1;
  while (end - start > 1) {
    uint32_t middle = (end + start) / 2;
    uint_t node_start = nodes[middle].beginning;
    uint_t node_end = nodes[middle].end;
    if ( index >= node_start && index < node_end){
      return middle;
    } else if (index < node_start) {
      end = middle;
    } else if (index >= node_end) {
      start = middle;
    } else {
      printf("should not happen\n");
      // assert(false);
    }
  }
  if ( index >= nodes[start].beginning && index < nodes[start].end){
    return start;
  } else if ( index >= nodes[end].beginning && index < nodes[end].end) {
    return end;
  } else if (index >= nodes[nodes.size() - 1].end) {
      return nodes.size() - 1;
  } else {
    //printf("no containing node trying again\n");
    return find_contaning_node(index);
  }
}

// insert elem at index
// and releases it when it is done with it
void inline PMA::insert(uint_t index, uint32_t elem_dest, uint32_t elem_val, uint32_t src) {
  uint32_t * vals = (uint32_t *) edges.vals;
  uint32_t * dests = (uint32_t *) edges.dests;
  // always deposit on the left
  if (dests[index] == NULL_VAL) {
    // printf("added to empty\n");
    vals[index] = elem_val;
    dests[index] = elem_dest;
  } else {
    slide_right(index, vals, dests);
    vals[index] = elem_val;
    dests[index] = elem_dest;
  }
}

inline bool PMA::insert_dist(uint_t leaf_index) {
  uint32_t level = edges.H - 1;
  uint_t len = edges.logN * 2;
  uint_t node_index = find_node(leaf_index, len);
  double density_b = upper_density_bound[level];

  grab_locks_range(node_index, len);
  double density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density >= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      level--;
      density_b = upper_density_bound[level];

      release_locks_range(node_index, len/2);
      node_index = find_node(node_index, len);

      grab_locks_range(node_index, len);
      density = get_density(&edges, node_index, len);
    } else {
      printf("error double listed when it shouldn't have \n");
      return false;
    }
  }

  redistribute(node_index, len);
  release_locks_range(node_index, len);
  return true;
}

inline bool PMA::insert_dist_serial(uint_t leaf_index) {
  uint32_t level = edges.H - 1;
  uint_t len = edges.logN * 2;
  uint_t node_index = find_node(leaf_index, len);
  double density_b = upper_density_bound[level];
  double density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density >= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      level--;
      density_b = upper_density_bound[level];
      node_index = find_node(node_index, len);
      density = get_density(&edges, node_index, len);
    } else {
      printf("error double listed when it shouldn't have \n");
      return false;
    }
  }

  redistribute(node_index, len);
  return true;
}

/* inline bool PMA::insert_dist_batch() {
    std::vector<std::pair<uint_t, uint_t>> rebalance_ranges;

    for (uint_t i = 0; i < edges.leaf_num; ++i) {
        if (edges.dist_flag[i]) {
            uint32_t level = edges.H;
            uint_t len = edges.logN;
            uint_t node_index = find_node(i * edges.logN, len);
            double density_b = upper_density_bound[level];
            double density = get_density_with_temp(&edges, node_index, len);

            while (density >= density_b) {
                len *= 2;
                if (len <= edges.N) {
                    level--;
                    density_b = upper_density_bound[level];
                    node_index = find_node(node_index, len);
                    density = get_density_with_temp(&edges, node_index, len);
                } else {
                    return false;
                }
            }
            rebalance_ranges.emplace_back(node_index, len);
        }
    }
    
    std::sort(rebalance_ranges.begin(), rebalance_ranges.end());

    std::vector<std::pair<uint_t, uint_t>> non_overlap_ranges;
    for (const auto& rng : rebalance_ranges) {
      if (non_overlap_ranges.empty() || rng.first >= non_overlap_ranges.back().first + non_overlap_ranges.back().second) {
        non_overlap_ranges.push_back(rng);
      } else {
        auto& last = non_overlap_ranges.back();
        uint_t new_end = std::max(last.first + last.second, rng.first + rng.second);
        last.second = new_end - last.first;
      }
    }

    parlay::parallel_for(0, rebalance_ranges.size(), [&](size_t idx) {
        auto [node_index, len] = rebalance_ranges[idx];
        redistribute(node_index, len);
    });
    
    parlay::parallel_for (0, edges.leaf_num, [&](uint64_t i) {
        edges.dist_flag[i] = false;
    });
    return true;
} */

inline bool PMA::insert_dist_batch(uint32_t leaf_id) {
  uint_t len = edges.logN * 2;
  uint_t leaf_index = leaf_id * edges.logN;
  uint_t node_index = find_node(leaf_index, len);
  
  if (!edges.dist_flag[leaf_id]) {
    return true;
  }

  grab_locks_range(node_index, len);

  if (!edges.dist_flag[leaf_id]) {
    release_locks_range(node_index, len);
    return true;
  }

  uint32_t level = edges.H - 1;
  double density_b = upper_density_bound[level];
  double density = get_density_with_temp(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density >= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      release_locks_range(node_index, len/2);
      level--;
      density_b = upper_density_bound[level];
      node_index = find_node(node_index, len);
      grab_locks_range(node_index, len);
      density = get_density_with_temp(&edges, node_index, len);
    } else {
      printf("error doubel listed when it shouldn't have \n");
      return false;
    }
  }
  redistribute_with_temp(node_index, len);
  for (uint_t i = node_index / edges.logN; i < (node_index + len) / edges.logN; ++i) {
    edges.dist_flag[i] = false;
  }
  release_locks_range(node_index, len);
  return true;
}

inline bool PMA::remove_dist_lock(uint_t leaf_index) {
  uint32_t level = edges.H - 1;
  uint_t len = edges.logN * 2;
  uint_t node_index = find_node(leaf_index, len);
  double density_b = lower_density_bound[level];

  grab_locks_range(node_index, len);
  double density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density <= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      release_locks_range(node_index, len/2);
      level--;
      density_b = lower_density_bound[level];
      node_index = find_node(node_index, len);
      grab_locks_range(node_index, len);
      density = get_density(&edges, node_index, len);
    } else {
      printf("error half listed when it shouldn't have \n");
      return false;
    }
  }

  redistribute(node_index, len);
  release_locks_range(node_index, len);
  return true;
}

/* inline bool PMA::remove_dist_batch() {
    std::vector<std::pair<uint_t, uint_t>> rebalance_ranges;

    for (uint_t i = 0; i < edges.leaf_num; ++i) {
        if (edges.dist_flag[i]) {
            uint32_t level = edges.H;
            uint_t len = edges.logN;
            uint_t node_index = find_node(i * edges.logN, len);
            double density_b = lower_density_bound[level];
            double density = get_density(&edges, node_index, len);

            while (density <= density_b) {
                len *= 2;
                if (len <= edges.N) {
                    level--;
                    density_b = lower_density_bound[level];
                    node_index = find_node(node_index, len);
                    density = get_density(&edges, node_index, len);
                } else {
                    return false;
                }
            }
            rebalance_ranges.emplace_back(node_index, len);
        }
    }

    printf("Rebalance ranges size: %lu\n", rebalance_ranges.size());

    std::sort(rebalance_ranges.begin(), rebalance_ranges.end());

    std::vector<std::pair<uint_t, uint_t>> non_overlap_ranges;
    for (const auto& rng : rebalance_ranges) {
      if (non_overlap_ranges.empty() || rng.first >= non_overlap_ranges.back().first + non_overlap_ranges.back().second) {
        non_overlap_ranges.push_back(rng);
      } else {
        auto& last = non_overlap_ranges.back();
        uint_t new_end = std::max(last.first + last.second, rng.first + rng.second);
        last.second = new_end - last.first;
      }
    }
    printf("Non-overlapping ranges size: %lu\n", non_overlap_ranges.size());
    parlay::parallel_for(0, rebalance_ranges.size(), [&](size_t idx) {
        auto [node_index, len] = rebalance_ranges[idx];
        redistribute(node_index, len);
    });
    printf("Finished redistributing\n");
    parlay::parallel_for (0, edges.leaf_num, [&](uint64_t i) {
        edges.dist_flag[i] = false;
    });
    return true;
} */

inline bool PMA::remove_dist_batch(uint_t leaf_index) {
  uint32_t level = edges.H;
  uint_t len = edges.logN;
  uint_t node_index = find_node(leaf_index, len);
  double density_b = lower_density_bound[level];

  grab_locks_range(node_index, len);
  double density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density <= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      release_locks_range(node_index, len/2);
      level--;
      density_b = lower_density_bound[level];
      node_index = find_node(node_index, len);
      grab_locks_range(node_index, len);
      density = get_density(&edges, node_index, len);
    } else {
      release_locks_range(node_index, len);
      // printf("error half listed when it shouldn't have \n");
      return false;
    }
  }
  
  redistribute(node_index, len);
  release_locks_range(node_index, len);
  return true;
}

inline bool PMA::remove_dist_serial(uint_t leaf_index) {
  uint32_t level = edges.H - 1;
  uint_t len = edges.logN * 2;
  uint_t node_index = find_node(leaf_index, len);
  double density_b = lower_density_bound[level];
  double density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density <= density_b) {
    // printf("node_index = %d, desnsity = %f, density bound = %f, len = %d, worker = %lu\n", node_index, density, density_b.y, len, get_worker_num());
    len *= 2;
    if (len <= edges.N) {
      release_locks_range(node_index, len/2);
      level--;
      density_b = lower_density_bound[level];
      node_index = find_node(node_index, len);
      density = get_density(&edges, node_index, len);
    } else {
      // printf("error half listed when it shouldn't have \n");
      return false;
    }
  }

  redistribute(node_index, len);
  return true;
}

// remove elem at index
// and releases it when it is done with it
void inline PMA::remove(uint_t index, uint32_t elem_dest, uint32_t src) {
  slide_left(index, edges.vals, edges.dests);
  return;
}

template<class T> inline void Log(const __m256i & value) {
    const size_t n = sizeof(__m256i) / sizeof(T);
    T buffer[n];
    _mm256_storeu_si256((__m256i*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
    printf("\n");
}


void inline PMA::print_graph() {
  uint32_t num_vertices = nodes.size();
  for (uint32_t i = 0; i < num_vertices; i++) {
    // uint32_t matrix_index = 0;
    // if (nodes[i].beginning > nodes[i].end) {
      // printf("error in node %d, beginning %d, end %d\n", i, nodes[i].beginning, nodes[i].end);
      // printf("\n");
    // }
    printf("%u \n", nodes[i].num_neighbors);
    for (uint_t j = nodes[i].beginning + 1; j < nodes[i].end; j++) {
      if (edges.dests[j]!=NULL_VAL) {
        // while (matrix_index < edges.dests[j]) {
          // printf("000 ");
          // matrix_index++;
        // }
        printf("%03d ", edges.dests[j]);
        // matrix_index++;
      }
    }
    // for (uint32_t j = matrix_index; j < num_vertices; j++) {
      // printf("000 ");
    // }
    printf("\n");
  }
}

// add a node to the graph
void inline PMA::add_node() {
  node_t node;
  uint32_t len = nodes.size();
  edge_t sentinel;
  sentinel.dest = SENT_VAL; // placeholder
  sentinel.value = len;       // back pointer

  if (len > 0) {
    node.beginning = nodes[len - 1].end;
    if(node.beginning == edges.N - 1) {
      uint32_t volatile  * volatile dests = edges.dests;
      uint_t leaf = find_leaf(&edges, node.beginning);
      // moving the beginning of the node you are inserting left
      //TODO jump back to the last leaf and look forward from there
      if (nodes[len-1].num_neighbors == 0) {
        node.beginning = nodes[len - 1].beginning + 1;
      } else {
        while(dests[node.beginning - 1] == NULL_VAL && node.beginning != leaf /* && next_leaf(node.beginning, edges.loglogN) != node.beginning*/) {
          node.beginning -= 1;
        }
      }
    }
    node.end = node.beginning + 1;
    // fix previous node to set its end to your beginning
    nodes[len - 1].end = node.beginning;
  } else {
    node.beginning = 0;
    node.end = 1;
    // be special to the first one since we know it never moves do it doesn't need to look like a sentinal since it doesn't have to be fixed ever
    sentinel.value = NULL_VAL;
    sentinel.dest = 0;
  }
  node.num_neighbors = 0;
  nodes.push_back(node);
  uint_t loc = node.beginning;
  insert(loc, sentinel.dest, sentinel.value, nodes.size() - 1);
}

inline uint32_t PMA::binary_find_leaf(uint32_t begin_id, uint32_t end_id, uint32_t dest) {
    uint32_t *dests = edges.dests;
    uint32_t step = edges.logN;
    uint32_t l = begin_id + 1, r = end_id, ans = end_id + 1;
    uint32_t mid = l + (r - l) / 2;
    __builtin_prefetch ((void *)&dests[mid * step], 0, 3);
    while (l <= r) {
        uint32_t x = dests[mid * step];
        if (x > dest) {
            ans = mid;
            r = mid - 1;
            mid = l + (r - l) / 2;
            __builtin_prefetch ((void *)&dests[mid * step], 0, 3);
        } else {
            l = mid + 1;
            mid = l + (r - l) / 2;
            __builtin_prefetch ((void *)&dests[mid * step], 0, 3);
        }
    }
    if (ans <= end_id)
        return ans - 1;
    else
        return end_id;
}

inline uint_t PMA::binary_find_elem(uint_t begin_idx, uint_t end_idx, uint32_t dest) {
    uint32_t *dests = edges.dests;
    uint_t l = begin_idx, r = end_idx - 1, ans = end_idx;
    uint_t mid = l + (r - l) / 2;
    __builtin_prefetch ((void *)&dests[mid], 0, 3);
    while (l <= r) {
        if (dests[mid] >= dest) {
            ans = mid;
            r = mid - 1;
            mid = l + (r - l) / 2;
            __builtin_prefetch ((void *)&dests[mid], 0, 3);
        } else {
            l = mid + 1;
            mid = l + (r - l) / 2;
            __builtin_prefetch ((void *)&dests[mid], 0, 3);
        }
    }
    return ans;
}

// batch delete exclude merge
void inline PMA::build_from_edges(uint32_t *srcs, uint32_t *dests, uint8_t * pma_edges, uint64_t vertex_count, uint64_t edge_count, uint32_t* additional_degrees) {
  // step 1 : build a CSR
  uint_t* vertex_array = (uint_t*)calloc(vertex_count, sizeof(uint_t)); 
  uint64_t edges_for_pma = 0;
  for(uint64_t i = 0; i < edge_count; i++) {
    if(pma_edges[i]) {
      uint32_t s = srcs[i];
      additional_degrees[s]++;
      vertex_array[s]++;
      edges_for_pma++;
    }
  }

  printf("build from edges: edges for pma = %u\n", edges_for_pma);
  nodes[0].num_neighbors = additional_degrees[0];

  // do prefix sum to get top level of CSR into vertex_array
  for(uint32_t i = 1; i < vertex_count; i++) {
    vertex_array[i] += vertex_array[i-1];
    nodes[i].num_neighbors = additional_degrees[i];
  }

  // and get CSR edge array into pma_edge_array
  uint32_t* pma_edge_array = (uint32_t*)malloc(edges_for_pma * sizeof(uint32_t));
  uint64_t pma_edges_so_far = 0;
  for(uint64_t i = 0; i < edge_count; i++) {
    if(pma_edges[i]) {
      pma_edge_array[pma_edges_so_far] = dests[i];
      pma_edges_so_far++;
    }
  } 
  
  uint64_t pma_size = vertex_count + pma_edges_so_far;
  uint64_t new_N = 1;
  while (new_N < pma_size) { new_N *= 2; }

  edges.N = new_N;
  edges.non_null = pma_size;
  edges.loglogN = bsr_word(bsr_word(edges.N) + 1);
  edges.logN = (1 << edges.loglogN);
  edges.mask_for_leaf = ~(edges.logN - 1);
  edges.leaf_num = new_N / edges.logN;
  // assert(edges.logN > 0);
  edges.density_limit = ((double) edges.logN - 1)/edges.logN;
  edges.H = bsr_word(edges.leaf_num);
  for (uint32_t i = 0; i <= edges.H; i++) {
    upper_density_bound[i] = density_bound(&edges, i).y;
    lower_density_bound[i] = density_bound(&edges, i).x;
  }

  printf("PMA initial size = %lu, logN = %u, leaf num = %u\n", edges.N, edges.logN, edges.leaf_num);

  uint_t num_elts = vertex_count + pma_edges_so_far;
  uint32_t *space_vals = (uint32_t *)aligned_alloc(32, num_elts * sizeof(*(edges.vals)));
  uint32_t *space_dests = (uint32_t *)aligned_alloc(32, num_elts * sizeof(*(edges.dests)));

  // step 2: write the PMA at the front of the new array
  // TODO: can also make this parallel
  uint64_t position_so_far = 0;
  for(uint64_t i = 0; i < vertex_count; i++) {
    // first, write the sentinel
    if (i == 0) {
      space_dests[position_so_far] = 0;
      space_vals[position_so_far] = NULL_VAL;
    } else {
      space_dests[position_so_far] = SENT_VAL;
      space_vals[position_so_far] = i;
    }
    position_so_far++;
    // then write the edges
    // printf("pma degree of vertex %u = %u\n", i, additional_degrees[i]);
    for(uint64_t j = 0; j < additional_degrees[i]; j++) {
      if (i == 0) {
        space_dests[position_so_far] = pma_edge_array[j];
      } else {
        space_dests[position_so_far] = pma_edge_array[vertex_array[i - 1] + j];
      }

      space_vals[position_so_far] = 1;
      position_so_far++;
    }
  }
  // assert(num_elts == position_so_far);

  printf("starting copy into PMA\n");
  // step 3: redistribute
  uint_t num_leaves = new_N >> edges.loglogN;
  uint_t count_per_leaf = num_elts / num_leaves;
  uint_t extra = num_elts % num_leaves;

  edges.temp_edges.resize(num_leaves);
  uint32_t *new_dests = (uint32_t *)aligned_alloc(32, new_N * sizeof(*(edges.dests)));
  uint32_t *new_vals = (uint32_t *)aligned_alloc(32, new_N * sizeof(*(edges.vals)));
  LOCK *new_leaf_locks = (LOCK *)aligned_alloc(32, num_leaves * sizeof(*(edges.leaf_locks)));
  bool *new_dist_flag = (bool *)aligned_alloc(32, num_leaves * sizeof(*(edges.dist_flag)));

  parlay::parallel_for (0, new_N, [&](uint64_t i) {
  //for (uint32_t i = 0; i < new_N; i++) {
    new_vals[i] = 0; // setting to null
    new_dests[i] = NULL_VAL; // setting to null
  });

  parlay::parallel_for(0, num_leaves, [&](uint64_t i) {
  // for(uint_t i = 0; i < num_leaves; i++) {
    // how many are going to this leaf
    uint_t count_for_leaf = count_per_leaf + (i < extra);
    // start of leaf in output
    uint_t in = ((i) << edges.loglogN);
    // start in input
    uint_t j2 = count_per_leaf*i +min(i,extra);
    uint_t j3 = j2;
    for(uint_t k = in; k < count_for_leaf+in; k++) {
      // assert(j2 < num_elts);
      new_vals[k] = space_vals[j2];
      j2++;
    }
    for (uint_t k = in; k < count_for_leaf+in; k++) {
      new_dests[k] = space_dests[j3];
      if (new_dests[k]==SENT_VAL) {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = space_vals[j3];
        fix_sentinel(node_index, k);
      }
      j3++;
    } 
    new_leaf_locks[i].init();
    new_dist_flag[i] = false;
  });

  free(space_dests);
  free(space_vals);
  free(vertex_array);

  free((void*)edges.vals);
  edges.vals = new_vals;
  free((void*)edges.dests);
  edges.dests = new_dests;
  free((void*)edges.leaf_locks);
  edges.leaf_locks = new_leaf_locks;
  free((void*)edges.dist_flag);
  edges.dist_flag = new_dist_flag;
}

int inline PMA::add_edge_lock(uint32_t src, uint32_t dest, uint32_t value, uint32_t leaf_begin, uint32_t leaf_end, uint32_t leaf_id) {
  edges.leaf_locks[leaf_id].lock();

  // if (!edges.temp_edges[leaf_id].empty()) {
    // __builtin_prefetch(&edges.temp_edges[leaf_id][0], 0, 3);
  // }

  uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
  uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
  uint_t loc_to_add = binary_find_elem(left, right, dest);

  if (edges.dests[loc_to_add] == dest) {
    edges.vals[loc_to_add] = value;
    edges.leaf_locks[leaf_id].unlock();
    return 0;
  }
  
  for (auto & e : edges.temp_edges[leaf_id]) {
    if (e.loc == loc_to_add && e.dest == dest) {
      e.value = value;
      edges.leaf_locks[leaf_id].unlock();
      return 0;
    }
  }
        
  // uint32_t count = get_density_count(&edges, leaf_id * edges.logN, edges.logN);
  __sync_fetch_and_add(&nodes[src].num_neighbors, 1);
  uint32_t last_id = ((leaf_id + 1) << edges.loglogN) - 1;

  if (edges.dests[last_id] == NULL_VAL) {
    insert(loc_to_add, dest, 1, src);
    edges.leaf_locks[leaf_id].unlock();
    return 1;
  } else {
    edges.temp_edges[leaf_id].push_back({loc_to_add, dest, value}); 
    if (!edges.dist_flag[leaf_id]) {
      edges.dist_flag[leaf_id] = true;
      edges.leaf_locks[leaf_id].unlock();
      return -2;
    }
    edges.leaf_locks[leaf_id].unlock();
    return -1;
  } 
}

bool inline PMA::add_edge_serial(uint32_t src, uint32_t dest, uint32_t value) {
  while (true) {
    __builtin_prefetch(&nodes[src]);
      
    uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
    uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
    uint32_t leaf_id = (leaf_end != leaf_begin)
                      ? binary_find_leaf(leaf_begin, leaf_end, dest)
                      : leaf_begin;

    uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
    uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
    uint_t loc_to_add = binary_find_elem(left, right, dest);

    if (edges.dests[loc_to_add] == dest) {
      edges.vals[loc_to_add] = value;
      return false;
    }
        
    uint32_t count = get_density_count(&edges, leaf_id * edges.logN, edges.logN);
        
    if (count < edges.logN) {
      nodes[src].num_neighbors++;
      insert(loc_to_add, dest, 1, src);
      return true;
    } else {
      if (!insert_dist_serial(leaf_id * edges.logN)) {
        exit(1);
      }
      continue;
    }  
  }
}

bool inline PMA::add_edge_once(uint32_t src, uint32_t dest, uint32_t value) {
  __builtin_prefetch(&nodes[src]);

  uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
  uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
  uint32_t leaf_id = (leaf_end != leaf_begin)
                      ? binary_find_leaf(leaf_begin, leaf_end, dest)
                      : leaf_begin;

  edges.leaf_locks[leaf_id].lock();
  uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
  uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
  uint_t loc_to_add = binary_find_elem(left, right, dest);

  if (edges.dests[loc_to_add] == dest) {
    edges.vals[loc_to_add] = value;
    edges.leaf_locks[leaf_id].unlock();
    return false;
  }

  for (auto & e : edges.temp_edges[leaf_id]) {
    if (e.dest == dest) {
      e.value = value;
      edges.leaf_locks[leaf_id].unlock();
      return false;
    }
  }
        
  uint32_t count = get_density_count(&edges, leaf_id * edges.logN, edges.logN);
  if (count < edges.logN) {
    __sync_fetch_and_add(&nodes[src].num_neighbors, 1);
    insert(loc_to_add, dest, 1, src);
    edges.leaf_locks[leaf_id].unlock();
    return true;
  } else {
    edges.temp_edges[leaf_id].push_back({loc_to_add, dest, value});
    edges.dist_flag[leaf_id] = true;
    edges.leaf_locks[leaf_id].unlock();
    return true;
  }
}

// do merge or not based on size of batch
void inline PMA::add_edge_batch_wrapper(pair_uint *es, uint64_t edge_count) {
    __builtin_prefetch(&edges, 0, 3);

    uint64_t max_non_null = edges.non_null + edge_count;
    while (max_non_null >= edges.N * upper_density_bound[0]) {
     double_list();
    }
    
    if (edge_count <= 100) {
      // printf("tiny batch inserting\n");
      for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t src = es[i].x;
        uint32_t dest = es[i].y;
        edges.non_null += add_edge_serial(src, dest, 1);
      }
      return;
    }

    // uint_t threshold = std::min(edges.leaf_num / 8, static_cast<uint32_t>(1000000000));
    uint_t threshold = 1000000000000;
    auto num_workers = parlay::num_workers();

    if (edge_count < threshold) {
      // printf("small batch inserting\n");

      // auto start = std::chrono::high_resolution_clock::now();

      parlay::sequence<uint_t> counts(num_workers, 0);
      std::vector<std::vector<uint32_t>> failed_leaf_ids(num_workers);
      bool dist = false;

      parlay::parallel_for(0, edge_count, [&](size_t i) {
      // for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t src = es[i].x;
        __builtin_prefetch(&nodes[src], 0, 3);
        uint32_t dest = es[i].y;
        uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
        uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
        uint32_t leaf_id = (leaf_end != leaf_begin)
                        ? binary_find_leaf(leaf_begin, leaf_end, dest)
                        : leaf_begin;
        int ret = add_edge_lock(src, dest, 1, leaf_begin, leaf_end, leaf_id);
        if (ret != 0) {
          counts[parlay::worker_id()] += 1;
          if (ret == -2) {
            dist = true;
            failed_leaf_ids[parlay::worker_id()].push_back(leaf_id);
          }
        }
      // }
      });
      uint_t true_count = parlay::reduce(counts);
      edges.non_null += true_count;

      // auto end1 = std::chrono::high_resolution_clock::now();
      // double elapsed1 = std::chrono::duration<double>(end1 - start).count();
      // printf("Batch add time: %.6f seconds\n", elapsed1);

      // TODO: redistrubute
      // uint32_t num = 0;
      if (dist == true) {
        // for (size_t i = 0; i < num_workers; i++) {
        parlay::parallel_for(0, num_workers, [&](size_t i) {
          for (auto & leaf_id : failed_leaf_ids[i]) {
            // __sync_fetch_and_add(&num, 1);
            if (!insert_dist_batch(leaf_id)) {
              exit(1);
            }
          }
        });
        // }
      }

      // auto end2 = std::chrono::high_resolution_clock::now();
      // double elapsed2 = std::chrono::duration<double>(end2 - end1).count();
      // printf("Batch dist time: %.6f seconds\n", elapsed2);
      // printf("number of redistributions = %u\n", num);

    } else {
      printf("large batch inserting\n");
      parlay::sequence<uint_t> counts(num_workers, 0);

      parlay::parallel_for(0, edge_count, [&](size_t i) {
      // for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t src = es[i].x;
        __builtin_prefetch(&nodes[src]);
        uint32_t dest = es[i].y;
        uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
        uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
        uint32_t leaf_id = (leaf_end != leaf_begin)
                        ? binary_find_leaf(leaf_begin, leaf_end, dest)
                        : leaf_begin;
        int ret = add_edge_lock(src, dest, 1, leaf_begin, leaf_end, leaf_id);
        if (ret != 0) {
          counts[parlay::worker_id()] += 1;
        }
      // }
      });
      uint_t true_count = parlay::reduce(counts);
      edges.non_null += true_count;

      parlay::parallel_for(0, edges.leaf_num, [&](size_t i) {
        if (edges.dist_flag[i]) {
          // __sync_fetch_and_add(&num, 1);
          if (!insert_dist_batch(i)) {
            exit(1);
          }
          // edges.dist_flag[i] = false;
        }
      });
      // }
      // printf("number of redistributions = %u\n", num);
      printf("done redistributing\n");
    }
}

// do merge or not based on size of batch
void inline PMA::remove_edge_batch_wrapper(pair_uint *es, uint64_t edge_count) {
    // uint32_t num_edges = 0;
    // for (uint32_t i = 0; i < nodes.size(); i++) {
      // num_edges += nodes[i].num_neighbors;
    // }
    // printf("num_edges = %u\n", num_edges);
    // printf("edges.non_null = %u \n", edges.non_null);

    __builtin_prefetch(&edges, 0, 3);

    if (edge_count <= 100) {
      // printf("small batch removing serially\n");
      for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t src = es[i].x;
        uint32_t dest = es[i].y;
        edges.non_null -= remove_edge_serial(src, dest, 1);
      }

      while (edges.non_null <= edges.N * lower_density_bound[0]) {
        half_list();
      }
      return;
    }

      // uint_t threshold = std::min(edges.leaf_num / 8, static_cast<uint32_t>(1000000));
      uint_t threshold = 10000000000000;
      auto num_workers = parlay::num_workers();
      parlay::sequence<uint_t> counts(num_workers, 0);

      if (edge_count < threshold) {
        // printf("small batch removing\n");
        std::vector<std::vector<uint32_t>> empty_leaf_ids(num_workers);
        bool dist = false;

        parlay::parallel_for(0, edge_count, [&](size_t i) {
          uint32_t src = es[i].x;
          __builtin_prefetch(&nodes[src]);
          uint32_t dest = es[i].y;
          uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
          uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
          uint32_t leaf_id = (leaf_end != leaf_begin)
                          ? binary_find_leaf(leaf_begin, leaf_end, dest)
                          : leaf_begin;
          int ret = remove_edge_lock(src, dest, 1, leaf_begin, leaf_end, leaf_id);
          if (ret != 0) {
            counts[parlay::worker_id()] += 1;
            if (ret == -1) {
              dist = true;
              empty_leaf_ids[parlay::worker_id()].push_back(leaf_id);
            }
          }
        });
        uint_t true_count = parlay::reduce(counts);
        edges.non_null -= true_count;
        
        parlay::parallel_for(0, num_workers, [&](size_t i) {
          for (auto & leaf_id : empty_leaf_ids[i]) {
            edges.dests[leaf_id * edges.logN] = NULL_VAL;
            edges.vals[leaf_id * edges.logN] = 0;
          }
        });

        bool half = false;
        while (edges.non_null <= edges.N * lower_density_bound[0]) {
          half = true;
          half_list();
        }

        // printf("finish halfing\n");
        
        if (half) {
          return;
        }
        
        // printf("start redistributing\n");

        if (dist == true) {
          parlay::parallel_for(0, num_workers, [&](size_t i) {
            for (auto & leaf_id : empty_leaf_ids[i]) {
              if (!remove_dist_batch(leaf_id)) {
                exit(1);
              }
              // edges.dist_flag[leaf_id] = false;
            }
          });
        }
        // printf("done redistributing\n");
      } else {
        printf("large batch removing\n");
        std::vector<uint32_t> loc_to_remove(edge_count);

        auto start = std::chrono::high_resolution_clock::now();

        parlay::parallel_for(0, edge_count, [&](size_t i) {
          uint32_t src  = es[i].x;
          uint32_t dest = es[i].y;
          uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
          uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
          uint32_t leaf_id = (leaf_end != leaf_begin)
                          ? binary_find_leaf(leaf_begin, leaf_end, dest)
                          : leaf_begin;
          uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
          uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
          loc_to_remove[i] = binary_find_elem(left, right, dest);
          if (edges.dests[loc_to_remove[i]] != dest) {
            loc_to_remove[i] = NULL_VAL;
          }
        });
        
        parlay::parallel_for(0, edge_count, [&](size_t i) {
          uint32_t src = es[i].x;
          counts[parlay::worker_id()] += remove_edge_once(src, loc_to_remove[i]);
        });
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        printf("Batch move time: %.6f seconds\n", elapsed);
        
        uint_t true_count = parlay::reduce(counts);
        edges.non_null -= true_count;

        bool half = false;
        while (edges.non_null <= edges.N * lower_density_bound[0]) {
          half = true;
          half_list();
        }
        
        if (half) {
          return;
        }

        parlay::parallel_for(0, edges.leaf_num, [&](size_t i) {
          if (edges.dist_flag[i]) {
            remove_dist_batch(i * edges.logN);
            edges.dist_flag[i] = false;
          }
        });

        printf("done redistributing\n");
      }
}

bool inline PMA::remove_edge_once(uint32_t src, uint32_t loc_to_remove) {
  // if (src == 111) {
    // printf("removing edge from node 111 at loc %u, dest = %u\n", loc_to_remove, edges.dests[loc_to_remove]);
  // }

  if (loc_to_remove == NULL_VAL) {
    return false;
  }
  uint32_t old_dest = edges.dests[loc_to_remove];
  if (old_dest != NULL_VAL) {
    bool success = __sync_bool_compare_and_swap(&edges.dests[loc_to_remove], old_dest, NULL_VAL);
    if (success) {
      edges.dests[loc_to_remove] = NULL_VAL;
      edges.vals[loc_to_remove] = 0;
      __sync_fetch_and_add(&nodes[src].num_neighbors, -1);
      edges.dist_flag[loc_to_remove >> edges.loglogN] = true;
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool inline PMA::remove_edge_serial(uint32_t src, uint32_t dest, uint32_t value) {
  __builtin_prefetch(&nodes[src]);
      
  uint32_t leaf_begin = (nodes[src].beginning) >> edges.loglogN;
  uint32_t leaf_end = (nodes[src].end) >> edges.loglogN;
  uint32_t leaf_id = (leaf_end != leaf_begin)
                      ? binary_find_leaf(leaf_begin, leaf_end, dest)
                      : leaf_begin;

  uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
  uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
  uint_t loc_to_remove = binary_find_elem(left, right, dest);

  if (edges.dests[loc_to_remove] != dest) {
    return false;
  }
      
  nodes[src].num_neighbors--;
  // printf("removing edge %u -> %u at loc %u\n", src, dest, loc_to_remove);
  remove(loc_to_remove, dest, src);

  uint32_t count = get_density_count(&edges, leaf_id * edges.logN, edges.logN);
        
  if (count == 0) {
    // remove_dist_serial(leaf_id * edges.logN);
    if (!remove_dist_serial(leaf_id * edges.logN)) {
      // exit(1);
    }
    return true;
  }
  return true;
}

int inline PMA::remove_edge_lock(uint32_t src, uint32_t dest, uint32_t value, uint32_t leaf_begin, uint32_t leaf_end, uint32_t leaf_id) {
    edges.leaf_locks[leaf_id].lock();
    uint_t left = (leaf_id == leaf_begin) ? nodes[src].beginning + 1 : leaf_id * edges.logN;
    uint_t right = (leaf_id == leaf_end) ? nodes[src].end : (leaf_id + 1) * edges.logN;
    uint_t loc_to_remove = binary_find_elem(left, right, dest);

    if (edges.dests[loc_to_remove] != dest || edges.vals[loc_to_remove] == NULL_VAL) {
      edges.leaf_locks[leaf_id].unlock();
      return 0;
    }
      
    __sync_fetch_and_add(&nodes[src].num_neighbors, -1);
    // uint32_t count = get_density_count(&edges, leaf_id * edges.logN, edges.logN);
    uint32_t second_id = (leaf_id << edges.loglogN) + 1;

    if (edges.dests[second_id] != NULL_VAL) {
      // print_array(leaf_id * edges.logN, edges.logN);
      // printf("removing edge %u -> %u at loc %u\n", src, dest, loc_to_remove);
      remove(loc_to_remove, dest, src);
      edges.leaf_locks[leaf_id].unlock();
      return 1; 
    } else {
      // print_array(leaf_id * edges.logN, edges.logN);
      edges.vals[loc_to_remove] = NULL_VAL;
      edges.leaf_locks[leaf_id].unlock();
      return -1;
    } 
}

void inline PMA::remove_edge_fast(uint32_t src, uint32_t dest) {
  __builtin_prefetch(&nodes[src], 0, 3);
  node_t node = nodes[src];
  uint_t loc_to_remove;
  uint32_t leaf_begin = find_leaf(&edges, node.beginning) / edges.logN;
  uint32_t leaf_end = find_leaf(&edges, node.end) / edges.logN;
  uint32_t leaf_id;

  if (leaf_end != leaf_begin) {
    leaf_id = binary_find_leaf(leaf_begin, leaf_end, dest);
    uint32_t left = (leaf_id == leaf_begin) ? node.beginning + 1 : leaf_id * edges.logN;
    uint32_t right = (leaf_id == leaf_end) ? node.end : (leaf_id + 1) * edges.logN;
    loc_to_remove = binary_find_elem(left, right, dest);
  } else {
    loc_to_remove = binary_find_elem(node.beginning + 1, node.end, dest);
  }
  
  if (edges.dests[loc_to_remove] != dest) {
    return;
  }
  __sync_fetch_and_add(&nodes[src].num_neighbors, -1);
  // printf("removing edge %u -> %u at loc %u\n", src, dest, loc_to_remove);
  remove(loc_to_remove, dest, src);
}

inline PMA::PMA(uint32_t init_n) {
  //making sure logN is at least 4
  full_num = init_n*(init_n - 1)/2 + init_n;
  edges.N = max(2UL << bsr_word(init_n*2), 16UL);
  edges.non_null = 0;
  edges.loglogN = bsr_word(bsr_word(edges.N) + 1);
  edges.logN = (1 << edges.loglogN);
  edges.mask_for_leaf = ~(edges.logN - 1);
  edges.leaf_num = edges.N / edges.logN;
  edges.temp_edges.resize(edges.leaf_num);

  // assert(edges.logN > 0);
  edges.density_limit = ((double) edges.logN - 1)/edges.logN;
  edges.H = bsr_word(edges.leaf_num);
  for (uint32_t i = 0; i <= edges.H; i++) {
    upper_density_bound[i] = density_bound(&edges, i).y;
    lower_density_bound[i] = density_bound(&edges, i).x;
  }

  edges.vals = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.vals)));
  edges.dests = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.dests)));
  edges.leaf_locks = (LOCK *)aligned_alloc(32, edges.leaf_num * sizeof(*(edges.leaf_locks)));
  edges.dist_flag = (bool *)aligned_alloc(32, edges.leaf_num * sizeof(*(edges.dist_flag)));

  temp_pfor (uint_t i = 0; i < edges.N; i++) {
    edges.vals[i] = 0;
    edges.dests[i] = NULL_VAL;
  }

  temp_pfor (uint_t i = 0; i < edges.leaf_num; i++) {
    edges.leaf_locks[i].init();
    edges.dist_flag[i] = false;
  }
  //TODO might be an issue if we grow it one at a time and let nodes be moved during operation
  nodes.reserve(init_n);
  for (uint32_t i = 0; i < init_n; i++) {
    add_node();
  }
}

inline PMA::PMA(PMA &other) {
  nodes = other.nodes;
  edges.temp_edges = other.edges.temp_edges;
  edges.N = other.edges.N;
  edges.non_null = other.edges.non_null;
  edges.loglogN = other.edges.loglogN;
  edges.logN = other.edges.logN;
  edges.mask_for_leaf = other.edges.mask_for_leaf;
  edges.density_limit = other.edges.density_limit;
  edges.H = other.edges.H;
  edges.leaf_num = other.edges.leaf_num;
  for (uint32_t i = 0; i <= edges.H; i++) {
    upper_density_bound[i] = other.upper_density_bound[i];
    lower_density_bound[i] = other.lower_density_bound[i];
  }

  edges.vals = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.vals)));
  memcpy(__builtin_assume_aligned((void *)edges.vals,32), __builtin_assume_aligned((void *)other.edges.vals, 32), edges.N * sizeof(*(edges.vals)));

  edges.dests = (uint32_t *)aligned_alloc(32, edges.N * sizeof(*(edges.dests)));
  memcpy(__builtin_assume_aligned((void *)edges.dests, 32), __builtin_assume_aligned((void *)other.edges.dests, 32), edges.N * sizeof(*(edges.dests)));

  // TODO: copy locks?

  // TODO: copy last dist time?
}

inline PMA::~PMA() { 
  free((void*)edges.vals);
  free((void*)edges.dests);
  free((void*)edges.leaf_locks);
  free((void*)edges.dist_flag);
}

inline void PMA::grab_locks_range(uint_t begin, uint_t len) {
  bool success = false;
  while (!success) {
    uint32_t locks_num = 0;
    if (len > edges.N) {
      printf("error grab locks range too big\n");
      // exit(1);
    }
    for (uint_t i = begin; i < begin + len; i += edges.logN) {
      uint32_t leaf_id = i / edges.logN;
      if (!edges.leaf_locks[leaf_id].try_lock()) {
        release_locks_range(begin, i - begin);
        usleep(1); // sleep 1 microsecond
        break;
      } else {
        locks_num++;
      }
    }

    if (locks_num == len / edges.logN) {
      success = true;
    }
  }
}

/* inline void PMA::try_grab_locks_range(uint_t begin, uint_t len) {
  for (uint_t i = begin; i < begin + len; i += edges.logN) {
    uint32_t leaf_id = i / edges.logN;
    if (!edges.leaf_locks[leaf_id].try_lock()) {
      release_locks_range(begin, i - begin);
      return;
    } else {
      locks_num++;
    }
  }
} */

inline void PMA::release_locks_range(uint_t begin, uint_t len) {
  for (uint_t i = begin; i < begin + len; i += edges.logN) {
    uint32_t leaf_id = i / edges.logN;
    edges.leaf_locks[leaf_id].unlock();
  }
}

}
