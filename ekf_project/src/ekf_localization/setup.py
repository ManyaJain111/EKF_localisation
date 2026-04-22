from setuptools import setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='EKF localization node',
    license='TODO',
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_localization.ekf_node:main',
        ],
    },
)