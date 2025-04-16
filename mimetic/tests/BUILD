load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "mimetic_integration_test",
    srcs = ["mimetic_integration_test.py"],
    deps = [
        "//mimetic:dataset",
        "//mimetic:utils",
        "//mimetic/model:event",
        "//mimetic/model:decision",
        "//mimetic/model:builder", 
        "//mimetic/model:system",
    ],
)