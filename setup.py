from setuptools import setup, find_packages

setup(
    name="delft",
    version="0.3.2",
    author="Patrice Lopez",
    author_email="patrice.lopez@science-miner.com",
    description="a Deep Learning Framework for Text",
    long_description=open("Readme.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kermitt2/delft",
    packages=find_packages(exclude=['test', '*.test', '*.test.*']),  
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'numpy==1.22.3',
        'regex==2021.11.10',
        'scikit-learn==1.0.1',
        'tqdm==4.62.3',
        'tensorflow==2.7.2',
        'h5py==3.6.0',
        'unidecode==1.3.2',
        'pydot==1.4.0',
        'lmdb==1.2.1',
        'transformers==4.15.0', 
        'torch==1.10.1',
        'truecase',
        'requests',
        'pandas==1.3.5',
        'pytest',
        'tensorflow-addons==0.15.0'  
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
