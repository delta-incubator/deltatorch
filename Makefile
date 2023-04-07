env:
	  sh ./scripts/create_venv.sh

test:
	source .venv/bin/activate && pip install -e . && pytest tests


clean:
	rm -rf *.egg-info && rm -rf .pytest_cache

format:
	black .

lint:
	prospector   --profile prospector.yaml && black --check lendingclub_scoring
