"""Provides the repository macro to import TensorRT Open Source components."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name = "tensorrt_oss_archive"):
    """Imports TensorRT Open Source Components."""
    TRT_OSS_COMMIT = "60c59af47e661a905cc5ded6272bae14e348dff1"
    TRT_OSS_SHA256 = "60fb69149ebb39299cd2387e57d29fadd1768074ddda0969c0141f50e2838f64"

    tf_http_archive(
        name = name,
        sha256 = TRT_OSS_SHA256,
        strip_prefix = "TensorRT-{commit}".format(commit = TRT_OSS_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            # TODO (wraveane): Restore the github URL when updated archive is published
            # "https://github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            "file:///fs/tmp/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
        ],
        build_file = "//third_party/tensorrt/plugin:BUILD",
        patch_file = ["//third_party/tensorrt/plugin:tensorrt_oss.patch"],
    )
