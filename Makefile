lint:
	black src run.py
	flake8 src run.py
	isort src run.py
