import logging
from typing import Any, Dict, Iterator, List

import more_itertools
from elasticsearch import Elasticsearch, ElasticsearchException
from elasticsearch.helpers import bulk

from interface.models import HmaoNpaDataset
from interface.text_features import extract_gosts

logger = logging.getLogger(__name__)


def create_index(index_name: str, es_client: Elasticsearch) -> None:
    mapping: Dict = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "npa_number": {"type": "keyword"},
                "dt": {"type": "date"},
                "gosts": {"type": "keyword"}
            }
        }
    }

    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index {index_name}")


def load(chunks: List[str]) -> Iterator[Any]:
    for chunk in chunks:
        try:
            yield generate_document_source(chunk)
        except Exception:
            raise


def generate_document_source(chunk: str) -> Dict:
    return {"text": chunk, "gosts": extract_gosts(chunk)}


def update_search(doc: HmaoNpaDataset, chunks: List[str], es_client: Elasticsearch, batch_size: int = 500) -> None:
    total_inserted_docs: int = 0
    total_errors: int = 0

    for chunk in more_itertools.ichunked(load(chunks), batch_size):
        bucket_data = []
        for document in chunk:
            cur = {
                "_index": "chunks",
                "_source": document,
            }
            cur['_source']['dt'] = doc.dt.isoformat()
            cur['_source']['npa_number'] = doc.npa_number
            bucket_data.append(cur)
        try:
            inserted, errors = bulk(es_client, bucket_data, max_retries=4, raise_on_error=False)
            errors_num = len(errors) if isinstance(errors, list) else errors  # type: ignore
            logger.debug(f"{inserted} docs successfully inserted by bulk with {errors_num} errors")
            total_inserted_docs += inserted
            total_errors += errors_num
            if isinstance(errors, list):  # type: ignore
                for error in errors:  # type: ignore
                    logger.error(f"Doc was not inserted with error: {error}")
        except ElasticsearchException as e:
            logger.exception(f"Error while pushing data to elasticsearch: {e}")
            raise


def search(es_client: Elasticsearch, config: Dict, question: str) -> List[str]:
    k = config["retrieval"]["top_k_fulltext"]
    query: Dict = {"query": {"match": {"text": {"query": question}}}, "size": k}
    response: Dict = es_client.search(index=config["elastic_params"]["index_name"], body=query)

    return [hit["_source"]["text"] for hit in response['hits']['hits']]
