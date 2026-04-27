from setuptools import find_packages, setup

import mser


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def parse_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return f.readlines()


setup(
    name="mser",
    packages=find_packages(),
    author="yeyupiaoling",
    version=mser.__version__,
    install_requires=parse_requirements(),
    description="Speech emotion recognition toolkit based on emotion2vec",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yeyupiaoling/SpeechEmotionRecognition-Pytorch",
    download_url="https://github.com/yeyupiaoling/SpeechEmotionRecognition-Pytorch.git",
    keywords=["audio", "emotion2vec", "speech"],
    entry_points={"console_scripts": ["mser-speech=mser.cli:main"]},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
    ],
    license="Apache License 2.0",
)
