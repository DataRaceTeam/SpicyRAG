run:
	docker-compose up

start:
	docker-compose up -d

tests:
	. $(VENV)/bin/activate && pytest --disable-warnings

clean:
	rm -rf __pycache__ .pytest_cache

fmt:
	sh -c "ruff format ."

down:
	docker-compose down
