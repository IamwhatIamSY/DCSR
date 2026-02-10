cc_library(
    name = "util",
    hdrs = ["util.h"],
)

cc_library(
  name = "partition",
  hdrs = ["partition.h"],
  deps = [
    "lock",
    "@ParallelTools//ParallelTools:parallel",
    "@ParallelTools//ParallelTools:reducer",
    "@parlaylib//parlay:primitives",
  ],
)

cc_library(
  name = "lock",
  hdrs = ["lock.h"],
  deps = [
    "util"
  ],
)

package(
    default_visibility = ["//visibility:public"],
)
