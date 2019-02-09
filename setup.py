import setuptools 

setuptools.setup(
    name="delft",
    version="0.1.0",
    author="Patrice Lopez",
    author_email="patrice.lopez@science-miner.com",
    description="a Deep Learning Framework for Text",
    long_description=open("Readme.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kermitt2/delft",
    packages=setuptools.find_packages(exclude=['test', '*.test', '*.test.*']),  
    include_package_data=True,
    python_requires='>=3.5',
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
