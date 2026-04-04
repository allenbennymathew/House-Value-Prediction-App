from invoke import task

@task
def complexity(c):
    """Run radon to measure code complexity."""
    print("Running Radon Cyclomatic Complexity evaluation...")
    c.run("radon cc src/ scripts/ -a -nc")
    
    print("\nRunning Radon Maintainability Index evaluation...")
    c.run("radon mi src/ scripts/")
