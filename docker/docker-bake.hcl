variable "IMAGE_REPOSITORY" {
  default = "madiatorlabs/sageattention-wheel-builder"
}

variable "IMAGE_TAG" {
  default = "dev"
}

group "default" {
  targets = ["builder-cu128", "builder-cu130"]
}

target "builder-common" {
  context    = "."
  dockerfile = "docker/Dockerfile.builder"
  platforms  = ["linux/amd64"]
  args = {
    PYTHON_VERSION     = "3.12"
    NVCC_TARGETS       = "sm_80;sm_86;sm_89;sm_90a;sm_120"
    BUILD_VERSION      = "1.2.2.post1"
    PACKAGING_VERSION  = "25.0"
    SETUPTOOLS_VERSION = "80.9.0"
    WHEEL_VERSION      = "0.45.1"
  }
}

target "builder-cu128" {
  inherits = ["builder-common"]
  tags = ["${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu128"]
  args = {
    CUDA_VERSION         = "12.8"
    CUDA_VERSION_DASH    = "12-8"
    TORCH_VERSION        = "2.10.0+cu128"
    TORCH_CUDA_VERSION   = "12.8"
    TORCH_INDEX_SUFFIX   = "cu128"
  }
}

target "builder-cu130" {
  inherits = ["builder-common"]
  tags = ["${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu130"]
  args = {
    CUDA_VERSION         = "13.0"
    CUDA_VERSION_DASH    = "13-0"
    TORCH_VERSION        = "2.10.0+cu130"
    TORCH_CUDA_VERSION   = "13.0"
    TORCH_INDEX_SUFFIX   = "cu130"
  }
}
