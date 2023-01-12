from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.2.0'
DESCRIPTION = 'UPAFuzzySystems package for definition and simulation of Fuzzy Inference Systems for general and control applications.'
LONG_DESCRIPTION = 'UPAFuzzySystems library allows defining Fuzzy Inference Systems for different applications with continuous and discrete universes, it also deploys structures for simulation of fuzzy control with transfer functions and state space models.'

# Setting up
setup(
    name="UPAFuzzySystems",
    version=VERSION,
    author="Dr. Martin Montes Rivera (Universidad Politécnica de Aguascalientes)",
    author_email="<martin.montes@upa.edu.mx>",
    url='https://github.com/UniversidadPolitecnicaAguascalientes/UPAFuzzySystems/',
    description=DESCRIPTION,
    license_files = ('LICENSE',),
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'control','scikit-fuzzy'],
    keywords=['python', 'fuzzy logic', 'fuzzy control', 'fuzzy inference systems', 'artificial intelligence'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
    
)