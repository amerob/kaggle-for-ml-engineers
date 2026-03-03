# Contributing to Kaggle for ML Engineers

Thank you for your interest in contributing to **Kaggle for ML Engineers: Competitive Systems & Applied Architecture**! This document provides guidelines and instructions for contributing to this project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Documentation Guidelines](#documentation-guidelines)
- [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to a positive environment:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior:

- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [amerhussein@gmail.com](mailto:amerhussein@gmail.com). All complaints will be reviewed and investigated promptly and fairly.

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to see if the problem has already been reported
2. **Use the latest version** to verify the issue still exists
3. **Collect information** about the bug (error messages, environment, reproduction steps)

When submitting a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details**: OS, Python version, package versions
- **Code samples** or minimal reproduction cases
- **Error messages** and stack traces

Use the bug report template when creating an issue.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear, descriptive title**
- **Provide detailed description** of the proposed enhancement
- **Explain why** this enhancement would be useful
- **List possible alternatives** you've considered
- **Include mockups or examples** if applicable

### Contributing Code

We welcome contributions in the following areas:

- **Bug fixes** for existing functionality
- **New features** aligned with the project's goals
- **Performance improvements**
- **Documentation improvements**
- **Test coverage** improvements
- **Example notebooks** demonstrating techniques

### Contributing Documentation

Documentation improvements include:

- Fixing typos or clarifying existing content
- Adding examples or code snippets
- Improving explanations of complex concepts
- Translating content to other languages
- Adding visualizations or diagrams

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or pyenv)

### Setting Up Your Development Environment

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/kaggle-for-ml-engineers.git
cd kaggle-for-ml-engineers

# 3. Add upstream remote
git remote add upstream https://github.com/amerhussein/kaggle-for-ml-engineers.git

# 4. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify setup
pytest tests/ -v
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Checkout your main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main
```

---

## Coding Standards

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings, single quotes for dictionary keys
- **Imports**: Grouped as standard library, third-party, local; sorted alphabetically

### Code Formatting

We use automated tools to enforce code style:

```bash
# Format code with black
black src/ tests/ --line-length 100

# Check with flake8
flake8 src/ tests/ --max-line-length 100

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically check code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

### Documentation Strings

All public functions, classes, and modules must have docstrings:

```python
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    cv: BaseCrossValidator
) -> Tuple[Any, np.ndarray]:
    """Train a model with cross-validation.

    Args:
        X: Feature matrix with shape (n_samples, n_features)
        y: Target vector with shape (n_samples,)
        params: Model hyperparameters
        cv: Cross-validation strategy

    Returns:
        Tuple containing:
            - Trained model object
            - Out-of-fold predictions array

    Raises:
        ValueError: If X and y have mismatched lengths
        TypeError: If params is not a dictionary

    Example:
        >>> model, oof_preds = train_model(X, y, params, cv)
        >>> print(f"OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
    """
    ...
```

### Naming Conventions

- **Modules**: lowercase_with_underscores.py
- **Classes**: PascalCase
- **Functions/Methods**: lowercase_with_underscores
- **Constants**: UPPERCASE_WITH_UNDERSCORES
- **Private**: _leading_underscore
- **Protected**: _single_leading_underscore

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

def process_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    fill_value: Union[int, float, str] = 0
) -> pd.DataFrame:
    ...
```

---

## Testing Requirements

### Test Coverage

All new code must include tests. We aim for:

- **Minimum 80%** code coverage
- **100% coverage** for critical path code

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_features.py -v

# Run with markers
pytest tests/ -v -m "not slow"
```

### Test Structure

```python
import pytest
import numpy as np
import pandas as pd
from src.features.encoding import TargetEncoder


class TestTargetEncoder:
    """Test suite for TargetEncoder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'C'],
            'target': [1, 0, 1, 0, 1]
        })
        self.encoder = TargetEncoder(smoothing=1.0)

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""
        result = self.encoder.fit_transform(
            self.df[['category']],
            self.df['target']
        )
        assert result is not None
        assert len(result) == len(self.df)

    def test_transform_unseen_category(self):
        """Test handling of unseen categories during transform."""
        self.encoder.fit(self.df[['category']], self.df['target'])
        
        new_df = pd.DataFrame({'category': ['D']})
        result = self.encoder.transform(new_df)
        
        # Unseen categories should get global mean
        expected = self.df['target'].mean()
        assert np.isclose(result.iloc[0, 0], expected)

    @pytest.mark.parametrize("smoothing", [0.0, 1.0, 10.0])
    def test_smoothing_parameter(self, smoothing):
        """Test different smoothing values."""
        encoder = TargetEncoder(smoothing=smoothing)
        result = encoder.fit_transform(
            self.df[['category']],
            self.df['target']
        )
        assert not result.isna().any().any()
```

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit        # Fast unit tests
@pytest.mark.integration # Integration tests
@pytest.mark.slow        # Long-running tests
@pytest.mark.gpu         # GPU-required tests
```

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**: `git fetch upstream && git merge upstream/main`
2. **Run tests locally**: `pytest tests/ -v`
3. **Check code style**: `black src/ && flake8 src/`
4. **Update documentation**: Ensure docstrings and README are current
5. **Add tests**: For any new functionality
6. **Update CHANGELOG.md**: Document your changes

### PR Guidelines

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make focused commits**: Each commit should be a logical unit
3. **Write clear commit messages**:
   ```
   feat: Add support for GPU-accelerated target encoding

   - Implement cuDF-based target encoder
   - Add fallback to pandas for CPU-only systems
   - Include comprehensive tests
   ```

4. **Push to your fork**: `git push origin feature/your-feature-name`
5. **Create a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Screenshots/examples if applicable
   - Checklist of completed items

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly and professionally
4. **Squash commits** if requested
5. **Merge** once approved

---

## Documentation Guidelines

### Markdown Files

- Use ATX-style headers (`#` not underlines)
- Wrap lines at 100 characters
- Use fenced code blocks with language tags
- Include table of contents for long documents

### Code Documentation

- Every public API must have docstrings
- Include usage examples in docstrings
- Document parameters, return values, and exceptions
- Keep examples runnable and tested

### Diagrams

- Use Mermaid for flowcharts and diagrams
- Include alt text for accessibility
- Keep diagrams simple and focused

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Creating a Release

1. Update version in `__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag -a v1.2.3 -m "Release version 1.2.3"`
4. Push tag: `git push origin v1.2.3`
5. GitHub Actions will build and create release

---

## Questions?

If you have questions about contributing:

- **General questions**: Open a [Discussion](https://github.com/amerhussein/kaggle-for-ml-engineers/discussions)
- **Bug reports**: Open an [Issue](https://github.com/amerhussein/kaggle-for-ml-engineers/issues)
- **Security issues**: Email [amerhussein@gmail.com](mailto:amerhussein@gmail.com) directly

Thank you for contributing to Kaggle for ML Engineers!
