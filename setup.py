import os
import io
import setuptools

readme_path = "README.md"
long_description = ""
if os.path.exists(readme_path):
    with io.open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setuptools.setup(
    name="quick_ssm",
    version="0.1.0",
    author="Sam Blouir",
    author_email="scblouir@gmail.com",
    description="A library for quick_ssm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samblouir/quick_ssm",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "tqdm",
        "biopython",
        "accelerate",
        "numpy==1.26.4",
        "einops",
    	"birdie_rl @ git+https://github.com/samblouir/birdie.git"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
