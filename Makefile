update:
	  poetry update && poetry install

test:
	poetry run pytest tests


clean:
	rm -rf *.egg-info && rm -rf .pytest_cache

format:
	poetry run black .

lint:
	 poetry run flake8 \
 		--max-line-length=150 \
 		--require-plugins="flake8-bugbear pep8-naming flake8-pyproject"

