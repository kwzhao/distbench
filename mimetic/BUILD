load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":utils",
        "@pypi//attrs",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//networkx",
    ],
)

py_library(
    name = "distbench",
    srcs = ["distbench.py"],
    deps = [
        ":utils",
        ":system",
        ":method",
        "//distbench:traffic_config_py_proto",
        "//distbench:joint_distribution_py_proto",
        "//mimetic/model:decision",
        "@pypi//attrs",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@pypi//attrs",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        ":dataset",
        "//mimetic/model:decision",
        "@pypi//attrs",
        "@pypi//pandas",
    ],
)

py_library(
    name = "system",
    srcs = [],
    deps = [
        "//mimetic/model:system",
    ],
)

py_library(
    name = "method",
    srcs = [],
    deps = [
        "//mimetic/model:method",
    ],
)