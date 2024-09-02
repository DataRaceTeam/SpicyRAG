run:
	docker compose up

start:
	docker compose up -d

tests:
	. $(VENV)/bin/activate && pytest --disable-warnings

clean:
	rm -rf __pycache__ .pytest_cache

fmt:
	sh -c "ruff format ."

down:
	docker compose down

test-integration:
	docker compose up -d
	sleep 5
	docker exec -it spicyrag-interface-1 sh -c "pytest tests/test_ragas_evaluation.py --disable-warnings"
