import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="docdoc",
    version="0.0.33",
    author="Enno",
    author_email="hi.xiaolonghuang@gmail.com",
    description="A tool to handle documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/click-me/docdoc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'sentence_splitter==1.4',
        'nltk==3.4.5',
        'numpy==1.18.1',
        'bert-tensorflow==1.0.1',
        'spacy==2.2.4'
    ],
    include_package_data = True
)


# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*