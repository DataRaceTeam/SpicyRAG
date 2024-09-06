run:
	docker compose up --no-recreate

start:
	docker compose up -d

tests:
	. $(VENV)/bin/activate && pytest --disable-warnings

clean:
	rm -rf __pycache__ .pytest_cache

clean-docker:
	docker compose down --rmi all --volumes --remove-orphans

clean-volume:
	docker volume rm spicyrag_postgres_data

fmt:
	sh -c "ruff format ."

down:
	docker compose down

test-integration:
	docker compose up -d
	sleep 5
	docker exec -it spicyrag-interface-1 sh -c "pytest tests/test_ragas_evaluation.py --disable-warnings"
