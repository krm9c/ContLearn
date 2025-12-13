# Claude Code Preferences

## Code Organization

### Directory Structure
- Use `src/` layout for Python packages (PEP 420 compliant)
- Keep logical groupings: `models/`, `trainers/`, `data/`, `config/`, `utils/`
- Place executable scripts in `scripts/` directory
- Keep tests at project root in `tests/`
- Keep config/data files at root, code in `src/`

### File Management
- **Minimize new files** - prefer editing existing files over creating new ones
- **Consolidate related code** - group related functionality in single files
- **Split only when necessary** - only split files when they exceed ~500-800 lines AND have clear logical divisions
- Archive old code to `old/` rather than immediate deletion

## Code Quality

### Comments
- **Always comment new code** - especially code added by Claude
- Add inline comments explaining non-obvious logic
- Include docstrings for functions and classes
- Mark Claude-added sections with `# Added by Claude:` when significant

### Code Reuse
- **Maximize reuse** - look for existing utilities before writing new ones
- Use inheritance and mixins for shared behavior
- Extract common patterns into helper functions
- Avoid duplicating logic across files

## Import Patterns

```python
# Prefer absolute imports with package prefix
from package.module import Class

# Use relative imports within same module (for internal components)
from .submodule import helper
```

## Documentation

### Markdown Files
- **Do not create summary/documentation markdown files automatically**
- **Ask before creating** any new markdown or summary files
- Only create documentation when explicitly requested

### Code Documentation
- Prefer inline comments over separate documentation files
- Keep README.md updated for major changes only
- Document APIs in docstrings, not separate files

## Workflow

### Changes
- Make minimal, focused changes
- Avoid over-engineering or adding unrequested features
- Test changes before considering complete
- Update existing files rather than creating new ones

### Communication
- Ask before creating new files (especially markdown/docs)
- Confirm approach for significant refactoring
- Report what was changed concisely without generating report files
