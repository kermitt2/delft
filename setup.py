from setuptools import setup, find_packages

setup(
    name="delft",
    version="0.3.4",
    author="Patrice Lopez",
    author_email="patrice.lopez@science-miner.com",
    description="a Deep Learning Framework for Text",
    long_description=open("Readme.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kermitt2/delft",
    packages=find_packages(exclude=['test', '*.test', '*.test.*']),  
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
