from setuptools import setup, find_packages

ignored_pkgs = '-e .'

def get_requirements(filename):
    '''Reads the requirements file and returns a list of requirements'''

    requirements =[]
    with open(filename, 'r') as f:
        for line in f:
            requirements.append(line.strip())

    if ignored_pkgs in requirements:
        requirements.remove(ignored_pkgs)
        #[req for req in requirements if not req.startswith(ignored_pkgs)]

    return requirements
    # with open(filename, 'r') as f:
    #     return f.read().splitlines()

setup(
    name='my_package',
    version='0.1',
    author='Devanshu Khandal',
    author_email='devanshu.khandal.27@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)