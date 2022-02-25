from setuptools import setup, find_packages

requirements = []
with open("requirements.txt") as freq:
    for line in freq.readlines():
        requirements.append(line.strip())

setup(
    name="judge_tools",
    version="0.1.0",
    author="Fan-s",
    description="Sentence fluency and relevance detection tool",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
)
