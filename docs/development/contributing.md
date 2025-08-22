# Contributing Guide

We welcome contributions to the RAG System! This guide outlines the process for contributing, from setting up your development environment to submitting your changes.

## Table of Contents

1.  [Code of Conduct](#code-of-conduct)
2.  [How to Contribute](#how-to-contribute)
3.  [Development Setup](#development-setup)
4.  [Coding Standards](#coding-standards)
5.  [Testing](#testing)
6.  [Submitting Changes](#submitting-changes)
7.  [Issue Reporting](#issue-reporting)
8.  [Feature Requests](#feature-requests)

## 1. Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [email protected].

## 2. How to Contribute

There are many ways to contribute to the RAG System:

*   **Bug Reports**: Identify and report issues.
*   **Feature Requests**: Suggest new features or enhancements.
*   **Code Contributions**: Fix bugs, implement new features, improve existing code.
*   **Documentation**: Improve existing documentation or create new guides.
*   **Testing**: Help test new features or existing functionality.
*   **Community Support**: Answer questions and help other users.

## 3. Development Setup

To set up your local development environment, follow the steps outlined in the [Development Setup Guide](development-setup.md).

## 4. Coding Standards

To ensure code quality and consistency, please adhere to the project's coding standards.

**Relevant Documentation:**
*   [Coding Standards Guide](coding-standards.md)

## 5. Testing

All code contributions should be accompanied by appropriate tests. The project uses `pytest` for unit and integration testing.

**Relevant Documentation:**
*   [Testing Guide](testing.md)

## 6. Submitting Changes

Follow these steps to submit your code contributions:

1.  **Fork the Repository**: Fork the `rag-engine` repository on GitHub.
2.  **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
git clone https://github.com/<your-username>/rag-engine.git
cd rag-engine
    ```
3.  **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
git checkout -b feature/my-new-feature
    ```
4.  **Make Your Changes**: Implement your changes, ensuring they adhere to coding standards and are covered by tests.
5.  **Run Tests**: Before committing, run all tests to ensure no regressions are introduced.
    ```bash
uv run python -m pytest
    ```
6.  **Commit Your Changes**: Write clear and concise commit messages.
    ```bash
git commit -m "feat: Add new feature for X"
    ```
7.  **Push to Your Fork**: Push your branch to your forked repository.
    ```bash
git push origin feature/my-new-feature
    ```
8.  **Create a Pull Request (PR)**: Open a pull request from your branch to the `main` branch of the upstream repository. Provide a clear description of your changes and reference any related issues.

## 7. Issue Reporting

If you find a bug or have an issue, please report it on the GitHub Issues page. When reporting, please include:

*   A clear and concise description of the issue.
*   Steps to reproduce the behavior.
*   Expected behavior.
*   Actual behavior.
*   Screenshots or error logs if applicable.
*   Your environment details (OS, Python version, RAG System version).

## 8. Feature Requests

If you have an idea for a new feature or enhancement, please open a feature request on the GitHub Issues page. Describe the feature, its benefits, and any potential use cases.
