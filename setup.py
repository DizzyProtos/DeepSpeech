from setuptools.command.install import install
from distutils.errors import DistutilsExecError
from subprocess import check_call

from setuptools import setup, find_packages


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        check_call('cd /decoders/ && sh setup.sh'.split())
        install.run(self)  


setup(name='deepspeechSTT',
    version='1',
    packages=find_packages(),
#     cmdclass={
#         'install': PostInstallCommand,
#     },
    install_requires=[
        'matplotlib>=3.0.0',
        'scipy==1.2.1',
        'resampy>=0.2.2',
        'SoundFile>=0.9.0.post1',
        'python_speech_features>=0.0.1'
    ]
)
