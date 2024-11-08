$env:PYTHONPATH = (Get-Location).Path; pytest tests/
pytest -v tests/