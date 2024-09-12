# Data Race Team
## AI Product Hack 2024 / case 9

## RAG pipeline:
--------
1) Предоставленный набор документов хранится в полнотекстовом индексе и разбивается на чанки.
2) С помощью трансформера создаются эмбединги чанков.
3) Запрос пользователя получается по API и подается в LLM Llama 8b для рерайтинга. Полученные вопросы векторизируются.
4) С помощью knn (n_neighbours = 3) находятся релевантные документы.
5) Формируются корзинки (вопрос + контекст), оборачиваются в промпт и подаются в LLM Llama 405b.
6) Ответ передаем по API клиенту.

## Project Organization
------------

    ├── README.md          <- Описание проекта.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── data               <- датасет, эвал и векторная БД
    │
    ├── interface          <- Бэк-энд сервиса
    │   ├── config.yaml    <- гиперпараметры вынесли в отдельный файл для гибкости кода
    │   │
    │   ├── database.py    <- полнотекстовая БД 
    │   ├── main.py        <- API и logger 
    |   ├── models.py      <- датасет и чанкер
    |   ├── schemas.py     <- ?
    |   ├── utils.py       <- основной RAG pipeline
    |
    ├── streamlit          <- streamlit front
    |
    ├── tests
    │   ├── confest.py     <- pytest framework
    │   ├── test_ragas_evaluation.py   <- оценка с помощью RAGAS
    │   │
    │
    └── 

## Как запустить проект 
--------
#### Setup
1. Configure Environment Variables

Make sure to set the following environment variables in your .env file or in your shell session to properly configure the interaction with the language models:

#### LLM API base URL and API key
LLM_BASE_URL="https://api.example.com"
LLM_API_KEY="your-api-key-here"

#### Rewriter API base URL and API key
LLM_REWRITER_BASE_URL="https://rewriter.example.com"
LLM_REWRITER_API_KEY="your-rewriter-api-key-here"
