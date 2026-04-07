from setuptools import setup, find_packages
setup(
    name="medical-triage-env",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["pydantic>=2.0.0"],
    python_requires=">=3.10",
)
