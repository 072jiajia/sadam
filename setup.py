import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="sadam",
    version="0.0.1",
    description="Scalable Adam Optimizer",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/072jiajia/sadam",
    author="Jijia Wu",
    author_email="jijiawu.cs@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
)
