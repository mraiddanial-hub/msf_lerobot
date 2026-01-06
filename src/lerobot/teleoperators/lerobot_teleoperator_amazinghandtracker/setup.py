from setuptools import setup, find_packages

setup(
    name="lerobot_teleoperator_amazinghandtracker",
    version="0.0.1",
    description="LeRobot AmazinghandTracker integration",
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




