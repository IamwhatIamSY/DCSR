// Discription: dcsr graph header file, which defines the graph 
// API and the overrall structure.

#pragma once

#include "partition.h"
#include "gbbs/bridge.h"

#include <variant>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <omp.h>
#include <vector>
#include <iterator>
#include <stdlib.h>

#include <cmath>
#include <chrono>  

namespace dcsr {
// define the max number of vertices in the block
uint32_t BLOCK_SIZE_log2 = 22; 
uint32_t BLOCK_SIZE = (1 << BLOCK_SIZE_log2);
uint32_t mask = (1 << BLOCK_SIZE_log2) - 1;
using E = const gbbs::empty;

class dcsr {
  public:
    dcsr(uint32_t num_v); // create a graph with the given size (#num nodes)

    // dcsr(std::string prefix); // read graph from disk

    // ~dcsr();

    // void dcsr(std::string prefix); // write graph to disk
    void build_graph(uint32_t *srcs, uint32_t *dests, uint8_t * is_edges, uint_t edge_count);

    bool add_edge(const uint32_t src, const uint32_t dest, uint32_t ed);

    void add_edge_batch(pair_uint *es, uint_t edge_count);

    bool remove_edge(const uint32_t src, const uint32_t dest);

    void remove_edge_batch(pair_uint *es, uint_t edge_count);

    void insert_perblock_range(const auto *es, const auto &offsets);

    void remove_perblock_range(const auto *es, const auto &offsets);

    // void build_from_batch(uint32_t *srcs, uint32_t *dests, uint32_t vertex_count, uint32_t edge_count);
    
    // check for the existence of the edge
    // uint32_t is_edge(const uint32_t s, const uint32_t d);
    // get out neighbors of vertex s

    // BlockIndex<E>::ConstNeighborView neighbors(const uint32_t v) const;
    // get out degree of vertex v
    uint32_t degree(const uint32_t v) const;
    
    //uint_t get_size(void);

    // uint32_t get_max_degree(void) const;
    template <class F>
    void map_neighbors(uint32_t v, F &&f) const;

    template <class F>
    void map_neighbors_early_exit(uint32_t v, F &&f) const;

    uint_t get_num_edges(void) const;

    uint32_t get_num_vertices(void) const;
  
  private:
    std::vector<std::unique_ptr<BLOCK>> blocks;
    uint32_t num_vertices{0};
    uint_t num_edges{0};
    unsigned int BLOCK_NUM{0}; // number of blocks
};

inline dcsr::dcsr(uint32_t num_v) : num_vertices(num_v) {
    // get the number of blocks
    BLOCK_NUM = (num_v + mask) >> BLOCK_SIZE_log2; // 向上取整
    // BLOCK_NUM = 1;
    blocks.reserve(BLOCK_NUM);
    cout << "BLOCK_NUM: " << BLOCK_NUM << endl;
    // cout << "BLOCK_SIZE: " << (1 << BLOCK_SIZE_log2) << endl;
    for (uint32_t i = 0; i < BLOCK_NUM; i++) {
        blocks.push_back(std::make_unique<BLOCK>());
    }
    cout << "sizeof(BLOCK)" << sizeof(BLOCK) << endl;
    // cout << "sizeof(nodes)" << sizeof(blocks[0].nodes[0])*(1 << BLOCK_SIZE_log2)<< endl;
    cout << "sizeof(node)" << sizeof(node_t)<< endl;
    // // // printf("SACASmalloc_usable_size(pma_edges.items), %u \n", malloc_usable_size(blocks[0].pma_edges.items));
    // cout << "here1" << endl;
}

//template <typename E>
//inline dcsr<E>::~dcsr() {}

inline void dcsr::insert_perblock_range(const auto *es, const auto &offsets) {
    parlay::parallel_for(0, offsets.size() - 1, [&](size_t i) { 
      for (auto it = es + offsets[i]; it < es + offsets[i + 1]; ++it) {
      //parlay::parallel_for(offsets[i], offsets[i + 1], [&](size_t it) {
        uint32_t block_index = std::get<0>(*(es + it)) >> BLOCK_SIZE_log2;
        uint32_t src = std::get<0>(*(es + it)) & mask;
        uint32_t dest = std::get<1>(*(es + it));
        //blocks[block_index]->add_edge(src, dest, 1);
      }
    });
}

inline void dcsr::remove_perblock_range(const auto *es, const auto &offsets) {
    parlay::parallel_for(0, offsets.size() - 1, [&](size_t i) { 
      for (auto it = es + offsets[i]; it < es + offsets[i + 1]; ++it) {
        uint32_t block_index = std::get<0>(*it) >> BLOCK_SIZE_log2;
        uint32_t src = std::get<0>(*it) & mask;
        uint32_t dest = std::get<1>(*it);
        blocks[block_index]->remove_edge(src, dest);
      }
    });
}

inline void dcsr::add_edge_batch(pair_uint *es, uint_t edge_count) {
    // parlay::parallel_for(0, edge_count, [&](uint32_t i) {
    for (uint_t i = 0; i < edge_count; i++) {
        uint32_t block_index = es[i].x >> BLOCK_SIZE_log2;
        uint32_t src = es[i].x & mask;
        uint32_t dest = es[i].y;
        blocks[block_index]->add_edge(src, dest, 1);
    // });
    }
}

inline void dcsr::remove_edge_batch(pair_uint *es, uint_t edge_count) {
    parlay::parallel_for(0, edge_count, [&](uint32_t i) {
    //for (uint_t i = 0; i < edge_count; i++) {
        uint32_t block_index = es[i].x >> BLOCK_SIZE_log2;
        uint32_t src = es[i].x & mask;
        uint32_t dest = es[i].y;
        blocks[block_index]->remove_edge(src, dest);
    //}
    });
}

inline uint32_t dcsr::degree(const uint32_t v) const {
    uint32_t block_index = v >> BLOCK_SIZE_log2;
    uint32_t v_id = v & mask;
    uint32_t degree = blocks[block_index]->nodes[v_id].degree;
    return degree;
}

inline uint_t dcsr::get_num_edges(void) const {
	return num_edges;
}

inline uint32_t dcsr::get_num_vertices(void) const {
    return num_vertices;
}

inline void dcsr::build_graph(uint32_t *srcs, uint32_t *dests, uint8_t * is_edges, uint_t edge_count) {
    std::vector<std::vector<uint32_t>> block_srcs(BLOCK_NUM);
    std::vector<std::vector<uint32_t>> block_dests(BLOCK_NUM);
    std::vector<std::vector<uint8_t>> block_edges(BLOCK_NUM);
    
    for (uint_t i = 0; i < edge_count; i++) {
        uint32_t block_index = srcs[i] >> BLOCK_SIZE_log2;
        block_srcs[block_index].push_back(srcs[i] & ((1 << BLOCK_SIZE_log2) - 1));
        //uint32_t block_index = 0;
        //block_srcs[block_index].push_back(srcs[i]);
        block_dests[block_index].push_back(dests[i]);
        block_edges[block_index].push_back(is_edges[i]); 
    }

    //for (uint32_t block_index = 0; block_index < BLOCK_NUM; block_index++) {
    parlay::parallel_for(0, BLOCK_NUM, [&](uint32_t block_index) {
        gbbs::sequence<uint32_t> degrees(MAX_NODE_NUM, 0);
        blocks[block_index]->build_from_edges(block_srcs[block_index].data(), 
                         block_dests[block_index].data(), 
                         block_edges[block_index].data(), 
                         MAX_NODE_NUM, 
                         block_edges[block_index].size(), 
                         degrees.data());
    });
    // // printf("finish build_from_edges %d\n", BLOCK_NUM);

    //for (uint32_t block_index = 0; block_index < BLOCK_NUM; block_index++) {
        //printf("the subgraph in block %u: \n", block_index);
        //blocks[block_index]->print_graph();
    //}
}

template <class F>
inline void dcsr::map_neighbors(uint32_t v, F &&f) const{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    unsigned int block_index = v >> BLOCK_SIZE_log2;
    //unsigned int block_index = 0;
    unsigned int v_id = v & mask;
    //unsigned int v_id = v;
    auto& block = blocks[block_index];
    auto& degree = blocks[block_index]->nodes[v_id].degree;
    uint_t start = blocks[block_index]->nodes[v_id].beginning + 1;
    uint_t end = blocks[block_index]->nodes[v_id].end;
    E empty_weight = E();
    if (degree > 1024) {
        parlay::parallel_for(start, end, [&](uint32_t j) { 
            if (blocks[block_index]->pma_edges.dests[j] != NULL_VAL) {
                f(v, blocks[block_index]->pma_edges.dests[j], empty_weight);
            }
        });
    } else {
        for (uint_t j = start; j < end; j++) {           
            if (blocks[block_index]->pma_edges.dests[j] != NULL_VAL) {
                f(v, blocks[block_index]->pma_edges.dests[j], empty_weight);
            }
        }
    }    
}

template <class F>
inline void dcsr::map_neighbors_early_exit(uint32_t v, F &&f) const {
    unsigned int block_index = v >> BLOCK_SIZE_log2;
    //unsigned int block_index = 0;
    unsigned int v_id = v & mask;
    //unsigned int v_id = v;
    auto& block = blocks[block_index];
    auto& degree = blocks[block_index]->nodes[v_id].degree;
    uint_t start = blocks[block_index]->nodes[v_id].beginning + 1;
    uint_t end = blocks[block_index]->nodes[v_id].end;
    E empty_weight = E();
    if (degree > 1024) {
        parlay::parallel_for(start, end, [&](uint32_t j) {
            if (blocks[block_index]->pma_edges.dests[j] != NULL_VAL) {
                if (f(v, blocks[block_index]->pma_edges.dests[j], empty_weight)) {
                    return;
                }
            }
        });
    } else {
        for (uint_t j = start; j < end; j++) {           
            if (blocks[block_index]->pma_edges.dests[j] != NULL_VAL) {
                if (f(v, blocks[block_index]->pma_edges.dests[j], empty_weight)) {
                    return;
                }
            }
        }
    }  
}

} // namespace dcsr