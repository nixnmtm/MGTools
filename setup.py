from setuptools import setup

setup(
    name='MGTools',
    version='3.1',
    packages=['mgt'],
    package_dir={'': 'mgt'},
    url='https://github.com/nixnmtm/MGTools',
    license='Biosimulation Lab NCTU',
    author='Nixon Raj',
    author_email='nixnmtm@gmail.com',
    description='Molecular Graph Theory',

    install_requires=[
                        "numpy",
                        "pandas",
                        "scipy",
                        "networkx",
                        "matplotlib",
                        "seaborn",
                        "natsort"
                   ],

)
