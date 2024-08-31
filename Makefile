.PHONY: run start format

run:
	docker-compose up

start:
	docker-compose up -d

fmt:
	sh -c "ruff format ."

down:
	docker-compose down
