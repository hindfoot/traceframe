import setuptools

setuptools.setup(
    name='traceframe',
    version='0.1',
    description='distributed traces to data frames',
    author='snible@us.ibm.com',
    license='Apache 2.0',
    license_files = ('LICENSE',),
    # packages=setuptools.find_packages(),

    package_data={'': ['py.typed']},
    packages=[""]
)
