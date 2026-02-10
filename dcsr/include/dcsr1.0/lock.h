#include <atomic>
#include <unistd.h>
#include <stdio.h>
#include "util.h"

#pragma once
#define num_tries 1

class LOCK {
public:
  std::atomic<uint32_t> x;

  void init();
  void lock();
  bool try_lock();
  void unlock();
  bool check_unlocked();
  void shared_lock();
  void shared_unlock();
};

inline void LOCK::init() {
  x.store(0, std::memory_order_release);
}

// should only be used in sequntial areas
inline bool LOCK::check_unlocked() {
  uint32_t lock = x.load(std::memory_order_acquire);
  return lock == 0;
}

// exclusive lock
inline void LOCK::lock() {
  uint32_t expected = 0;
  while (!x.compare_exchange_weak(
            expected, 
            1, 
            std::memory_order_acquire,
            std::memory_order_relaxed))
  {
    expected = 0;
  }
}

inline void LOCK::shared_lock() {
  while (true) {
    uint32_t cur = x.load(std::memory_order_acquire);
    if (cur == 1) continue;
    if (x.compare_exchange_weak(cur, cur + 2, std::memory_order_acquire, std::memory_order_relaxed)) {
      break;
    }
  }
}

inline void LOCK::shared_unlock() {
  x.fetch_sub(2, std::memory_order_release);
}

inline bool LOCK::try_lock() {
  uint32_t expected = 0;
  for (int i = 0; i < num_tries; i++) {
    if (x.compare_exchange_weak(
      expected, 
      1, 
      std::memory_order_acquire,
      std::memory_order_relaxed))
    {
      return true;
    }
      expected = 0;
  }
  return false;
}

inline void LOCK::unlock() {
  
  x.store(0, std::memory_order_acquire);

}

