from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ConvCUDA",
    ext_modules=[
        CUDAExtension(
            "ConvCUDA",
            [
                "/".join(__file__.split("/")[:-1] + ["base.cpp"]),
                "/".join(__file__.split("/")[:-1] + ["conv.cu"]),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
