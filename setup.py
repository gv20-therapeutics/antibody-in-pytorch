# -*- coding: utf-8 -*-
from setuptools import setup

import re
version = re.search(
    "__version__\s*=\s*'(.*)'",
    open('AIPT/entry_point.py').read()
    ).group(1)

setup(
    name="AIPT",
    packages=["AIPT",
              "AIPT.Utils",
              "AIPT.Utils.Dev",
              "AIPT.Models",
              "AIPT.Models.Liu2019",
              "AIPT.Models.Mason2020",
              "AIPT.Models.Wollacott2019",
              "AIPT.Models.Beshnova2020",
              "AIPT.Benchmarks",
              "AIPT.Benchmarks.OAS_dataset",
              "AIPT.Benchmarks.Liu2019_enrichment"],
    entry_points={
        "console_scripts": ['AIPT=AIPT.entry_point:main']
    },
    version=version,
    license='MIT',
    description="Machine learning models for antibody sequences in PyTorch",
    long_description="""Recently, more people are realizing the use of machine learning, especially deep learning, in helping to understand antibody sequences in terms of binding specificity, therapeutic potential, and developability. Several models have been proposed and shown excellent performance in different datasets. We believe there should be an optional solution of modeling antibody sequences, because if otherwise, people can use transfer learning to keep the good "knowledge" and train a minimal amount of parameters for specific tasks. Therefore, we create this public repo to collect and re-implement (if needed) public available machine learning models in PyTorch.""",
    author="GV20 Therapeutics AI Team",
    author_email="xihao_hu@gv20therapeutics.com",
    install_requires=['torch', 'torchvision', 'matplotlib', 'pandas', 'scipy', 'scikit-learn', 'seaborn', 'ipython', 'jupyter'],
    url="https://github.com/gv20-therapeutics/antibody-in-pytorch/",
)
