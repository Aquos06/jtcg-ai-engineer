seed_db:
	python3 seed_data.py

attu:
	docker rm -f attu
	docker run -d -p 9090:3000 -e MILVUS_URL=0.0.0.0:19530 --name attu zilliz/attu:latest

setup:
	docker compose up -d