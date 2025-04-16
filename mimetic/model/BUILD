load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "model",
    deps = [
        ":arrival",
        ":builder",
        ":decision",
        ":event",
        ":method",
        ":system",
    ],
)

py_library(
    name = "arrival",
    srcs = ["arrival.py"],
    deps = [
        "//mimetic:dataset",
        "//mimetic:utils",
        "@pypi//attrs",
    ],
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    deps = [
        ":event",
        ":decision",
        "//mimetic:utils",
        "@pypi//attrs",
    ],
)

py_library(
    name = "decision",
    srcs = ["decision.py"],
    deps = [
        ":event",
        "//mimetic:utils",
        "@pypi//attrs",
    ],
)

py_library(
    name = "event",
    srcs = ["event.py"],
    deps = [
        "//mimetic:dataset",
        "@pypi//attrs",
    ],
)

py_library(
    name = "method",
    srcs = ["method.py"],
    deps = [
        "//mimetic:dataset",
        ":event",
        ":decision",
        ":builder",
        "@pypi//attrs",
    ],
)

py_library(
    name = "system",
    srcs = ["system.py"],
    deps = [
        "//mimetic:dataset",
        ":method",
        ":arrival",
        ":event",
        "@pypi//attrs",
        "@pypi//tqdm",
    ],
)