from distutils.core import setup

setup(
    name='wsid',
    version='0.0.3',
    packages=['wsid', 'tests'],
    url='',
    install_requires=[
        'nltk>=3.0.5',
        'numpy>=1.9.3',
        'scipy>=0.13.3',
        'scikit-learn>=0.17',
        'matplotlib>=1.4.3',
        'diskcache',
        'python-igraph'
    ],
    author='Artem Revenko (artreven)',
    author_email='artreven@gmail.com',
    description='Induction and disambiguation of word senses using cooccurent words'
)
