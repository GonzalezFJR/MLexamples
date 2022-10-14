# install with "pip install -e ."
import setuptools

setuptools.setup(
    name='MLtools',
    version='0.0.0',
    description='ML examples with pytorch',
    packages=setuptools.find_packages(),
    package_data={
        "MLexamples" : [
            "data/*",
        ],
    }
)

