import setuptools 

setuptools.setup(
    name="delft",
    version="0.2.3",
    author="Patrice Lopez",
    author_email="patrice.lopez@science-miner.com",
    description="a Deep Learning Framework for Text",
    long_description=open("Readme.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kermitt2/delft",
    packages=setuptools.find_packages(exclude=['test', '*.test', '*.test.*']),  
    include_package_data=True,
    python_requires='>=3.5',
    install_requires=[
        'keras==2.2.4',
        'numpy>=1.16.1',
        'pandas>=0.22.0',
        'bleach>=2.1.0',
        'regex>=2018.2.21',
        'scikit-learn>=0.19.1',
        'tqdm>=4.21',
        'tensorflow_gpu==1.12.0',
        'gensim>=3.4.0',
        'langdetect>=1.0.7',
        'textblob>=0.15.1',
        'h5py>=2.7.1',
        'unidecode>=1.0.22',
        'pydot>=1.2.4',
        'lmdb>=0.94',
        'keras-bert==0.39.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
