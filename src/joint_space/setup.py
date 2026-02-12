from setuptools import find_packages, setup

package_name = 'joint_space'

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
    maintainer='itachi',
    maintainer_email='surya.roboengr@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'trapezoidal_controller=joint_space.trapezoidal_controller:main',
            'motion_planner=joint_space.motion_planner:main',
        ],
    },
)
