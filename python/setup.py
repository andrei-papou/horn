import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='horn',
    version='0.0.1',
    author='Andrei Papou',
    author_email='popow.andrej2009@yandex.ru',
    description='Library for saving Keras models for future use in the Rust code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andrei-papou/horn',
    packages=setuptools.find_packages(),
    install_requires=[
        'keras==2.2.4',
        'scikit-learn==0.20.4',
        'tensorflow==1.15.2',
        'typing==3.7.4',
    ]
)
