from setuptools import find_packages, setup

package_name = 'my_turtlebot3_potential_fields'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kermit',
    maintainer_email='olibia-navarrete@hotmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'potential_fields_node = my_turtlebot3_potential_fields.potential_fields_node:main'
        ],
    },
)
