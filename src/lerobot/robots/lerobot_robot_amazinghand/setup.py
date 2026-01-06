from setuptools import setup, find_packages

setup(
    name="lerobot_robot_amazinghand",
    version="0.0.1",
    description="LeRobot Amazinghand integration",
    author="Raid",
    author_email="m.raiddanial@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "teleop",
        "lerobot",
        "opencv-python",
        "scipy",
        "numpy",
        "mediapipe",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
)




