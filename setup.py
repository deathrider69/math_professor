from setuptools import setup, find_packages

setup(
    name="math-professor-agent",
    version="2.0.0",
    description="LangGraph + DSPy Mathematical Professor System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "dspy-ai>=2.6.27",
        "langgraph>=0.2.60",
        "langgraph-checkpoint>=2.0.6",
        "google-generativeai>=0.8.5",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12.0",
        "chromadb>=1.0.15",
        "sentence-transformers>=5.0.0",
        "tavily-python>=0.7.9",
        "openai>=1.75.0",
        "python-dotenv>=1.1.1",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "math-professor=streamlit_math_app:main",
            "test-system=test_langgraph_system:test_system",
        ],
    },
)