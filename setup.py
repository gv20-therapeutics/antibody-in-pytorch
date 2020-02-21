import setuptools

setuptools.setup(name='bi_lstm',
                 version='1.0',
                 author='Xihao Hu',
                 author_email='huxihao@gmail.com',
                 license='MIT',
                 packages=[''],
                 install_requires=[
        'torch', 'torchvision', 'matplotlib', 'pandas', 'scipy', 
        'scikit-learn', 'seaborn', 'ipython', 'jupyter', 'lifelines'
      ],
                 zip_safe=False)

