# Contributing to DEMS

Thank you for your interest in contributing to the Dynamic Energy Management System!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/dems.git
   cd dems
   ```

3. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

## Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Code Standards

### Python Style
- Follow PEP 8
- Use 4 spaces for indentation
- Max line length: 100 characters
- Type hints recommended

### JavaScript/React
- Use Prettier for formatting
- Follow ESLint rules
- Functional components preferred
- Use hooks for state management

### Documentation
- Docstrings for all modules and functions
- Clear comments for complex logic
- Update README for major changes

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_energy_manager.py -v

# Check code style
flake8 src/
black src/ --check
mypy src/
```

## Testing Guidelines

- Write tests for new features
- Maintain >80% code coverage
- Test edge cases and errors
- Use descriptive test names

## Commit Guidelines

Use conventional commits:
```
feat: Add new feature
fix: Fix a bug
docs: Update documentation
test: Add tests
refactor: Refactor code
style: Format code
chore: Update dependencies
```

Example:
```
feat: Add multi-node load balancing algorithm

- Implements distributed load balancing
- Reduces power loss by 15%
- Closes #123
```

## Pull Request Process

1. Update documentation
2. Add tests for changes
3. Ensure all tests pass
4. Submit PR with clear description
5. Address review comments

## Reporting Issues

Use GitHub Issues with:
- Clear title
- Detailed description
- Steps to reproduce (if bug)
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## Feature Requests

Submit feature requests as GitHub Issues with:
- Use case description
- Why it's valuable
- Proposed implementation (optional)

## Code of Conduct

- Be respectful and inclusive
- Avoid offensive language
- Respect diverse opinions
- Focus on constructive feedback

## License

By contributing, you agree your code will be licensed under MIT License.

## Questions?

- Open a discussion on GitHub
- Check existing issues/PRs
- Review documentation

---

Happy contributing! 🚀
