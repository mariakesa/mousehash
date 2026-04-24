from setuptools import find_packages, setup

setup(
    name="mousehash",
    version="0.1.0",
    description="DataJoint-backed research workflow system for Allen stimulus representations",
    author="Maria Kesa",
    author_email="mariarosekesa@gmail.com",
    url="https://github.com/mariakesa/mousehash",
    install_requires=[
        "allensdk @ git+https://github.com/AllenInstitute/AllenSDK.git",
        "datajoint>=0.14",
        "scikit-learn>=1.2",
        "scipy>=1.11",
        "torch>=2.1",
        "pandas>=2.2,<3",
        "numpy>=1.26,<3",
        "transformers>=4.40,<5",
        "plotly>=5.9",
        "Pillow>=10",
        "pyarrow>=15",
        "python-dotenv>=1.0",
    ],
    extras_require={
        "agents": [
            "smolagents>=1.7.0",
        ],
        "dev": [
            "pytest>=8",
            "black>=24",
            "ruff>=0.5",
        ],
    },
    packages=find_packages("src"),
    package_dir={"": "src"},
)