# Data Race Team - AI Product Hack 2024 / case 9

## RAG pipeline:

1) Предоставленный набор документов хранится в полнотекстовом индексе и разбивается на чанки.
2) С помощью трансформера создаются эмбединги чанков.
3) Запрос пользователя получается по API и подается в LLM Llama 70b для формирования ответа. Полученный ответ векторизируется.
4) С помощью knn (n_neighbours = 3) находятся релевантные документы.
5) Формируются корзинки (вопрос + контекст), оборачиваются в промпт и подаются в LLM Llama 405b.
6) Ответ передается по API клиенту.

## Структура репозитория
------------

    ├── README.md                   <- описание проекта
    │
    ├── notebooks                   <- jupyter notebooks
    │
    ├── data                        <- датасет, эвал и векторная БД
    │
    ├── interface                   <- бэк-энд сервиса
    |   ├── chunker.py              <- чанкер
    │   ├── config.yaml             <- гиперпараметры вынесли в отдельный файл для гибкости кода.
        |                              можно управлять экспериментами, не погружаясь в кодовую базу 
    │   ├── database.py             <- полнотекстовая БД 
    |   ├── elastic.py              <- elasticsearch
    |   ├── embedder.py             <- user-base encoder
    │   ├── main.py                 <- API и logger 
    |   ├── models.py               <- датасет и чанкер
    |   ├── schemas.py              <- интерфейс API
    |   ├── text_features.py        <- регулярные выражения для поиска параметров в тексте 
    |   ├── text_transformation.py  <- предобработка текста
    |   ├── utils.py                <- основной RAG pipeline
    |
    ├── streamlit                   <- streamlit front
    |
    ├── tests                       <- оценка с помощью RAGAS
    │
    └── requirements.txt            <- требования для запуска проекта

--------

## Как запустить проект 

1. Клонируйте репозиторий:

Сначала клонируйте репозиторий на ваш компьютер с помощью команды:

    git clone <repository_url>
    cd <repository_directory>

2. Создайте файл .env:

В корневом каталоге проекта создайте файл .env и добавьте следующие переменные окружения:

    LLM_BASE_URL=<ваш_base_url>
    LLM_API_KEY=<ваш_api_key>

    LLM_REWRITER_BASE_URL=<ваш_base_url>
    LLM_REWRITER_API_KEY=<ваш_api_key>

Замените LLM_BASE_URL и LLM_API_KEY на ваши актуальные значения.

3. Настройте модель эмбеддера:

Откройте файл interface/config.yaml и установите значения для параметров embedding_model и dimension в соответствии с конфигурацией, указанной в файле data/embeddings/embeddings_config.yaml.

Файл interface/config.yaml позволяет менять параметры эксперимента, не поглужаясь в кодовую базу. 

 
4. Запустите проект:

Чтобы запустить проект, выполните следующую команду:

    make run



## Материалы 
Ниже представлена часть материалов, которые использоваи для подготовки проекта

https://habr.com/ru/articles/779526/ (RAG — простое и понятное объяснение)

https://www.rungalileo.io/blog/mastering-rag-how-to-architect-an-enterprise-rag-system (Mastering RAG: How To Architect An Enterprise RAG System)

https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex (RAG - Encoder and Reranker evaluation)

https://habr.com/ru/companies/raft/articles/791034/ (Архитектура RAG: полный гайд)

https://habr.com/ru/companies/raft/articles/818781/ (Архитектура RAG: часть вторая — Advanced RAG)

https://youtu.be/sVcwVQRHIc8?si=4TcxFYeGwnjVnfhG (RAG from scratch)

https://youtu.be/kEgeegk9iqo?si=boVl0jwyMI3qSsJ7 (Advanced RAG with colbert)
