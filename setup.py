from distutils.core import setup

setup(
    name='cloud-sensor',
    version='0.0.1',
    packages=['cloud_sensor','cloud_sensor.src', 'cloud_sensor.src.util', 'cloud_sensor.src.util.data'],
    package_data={'cloud_sensor.src.util.data': ['*.png']},
    url='https://github.com/tribeiro/cloud-sensor/',
    license='',
    author='Tiago Ribeiro de Souza',
    author_email='tribeiro@ufs.br',
    description='Analyse images of the sky and detect clouds.',
    install_requires=['astropy',
                      ],
    scripts=['scripts/cloud_stats']
)
