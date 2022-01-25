"""Provides the repository macro to import TensorRT Open Source components."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name = "tensorrt_oss_archive"):
    """Imports TensorRT Open Source Components."""
    TRT_OSS_COMMIT = "6f38570b74066ef464744bc789f8512191f1cbc0"
    TRT_OSS_SHA256 = "3b5f573488e15dc822b3b7b550f7022ca461c241f3f5bd657a80fb949cec2ed6"

    tf_http_archive(
        name = name,
        sha256 = TRT_OSS_SHA256,
        strip_prefix = "TensorRT-{commit}".format(commit = TRT_OSS_COMMIT),
        urls = [
            # TODO: Google Mirror "https://storage.googleapis.com/...."
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            "https://github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
        ],
        build_file = "//third_party/tensorrt/plugin:BUILD",
        patch_file = ["//third_party/tensorrt/plugin:tensorrt_oss.patch"],
    )
