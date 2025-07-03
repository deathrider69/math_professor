from setuptools import setup, find_packages

setup(
    name="math-professor-agent",
    version="1.0.0",
    description="Agentic-RAG Mathematical Professor System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "dspy-ai>=2.4.0",
        "google-generativeai>=0.3.0",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)