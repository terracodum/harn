# Benchmark Harness (MVP)

Автономный конвейер: **репозиторий + бриф → верифицированный бенчмарк-кейс**
(`task/` + `evidence/` + `result.json`). Архитектура и этапы описаны в
[HARNESS_ARCHITECTURE.md](HARNESS_ARCHITECTURE.md).

Стек MVP:

- LLM — любой **OpenAI-совместимый** endpoint (Ollama с `gpt-oss:120b` / `qwen3`, vLLM, GigaChat через OpenAI-прокси).
- Семантический поиск по коду — **локально через [zvec](https://pypi.org/project/zvec/)**: FTS по идентификаторам + опциональный dense-вектор с `/v1/embeddings`, слияние RRF. Без MCP. Если zvec не установлен — фолбэк на поиск по ключевым словам.
- Песочница — Docker, `--network none`, pytest + JUnit XML → `reward.txt`.

## Установка

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

## Настройка LLM

Через блок `llm` во входном JSON, флаги CLI или переменные окружения:

| Переменная            | Назначение                                   |
|-----------------------|----------------------------------------------|
| `LLM_BASE_URL`        | `http://localhost:11434/v1` (Ollama), vLLM…   |
| `LLM_MODEL`           | `gpt-oss:120b`, `qwen3:32b`, `GigaChat-Pro`   |
| `LLM_API_KEY`         | ключ (для Ollama можно не задавать)           |
| `LLM_EMBED_MODEL`     | `nomic-embed-text`, `bge-m3` (опционально)     |
| `LLM_EMBED_BASE_URL`  | если эмбеддинги на другом сервере             |
| `LLM_TRUST_ENV`       | `1` — уважать системный прокси (по умолчанию выключено: прокси Windows перехватывает localhost и отвечает 503) |

JSON ответ запрашивается в самом строгом режиме, который поддерживает сервер
(`json_schema` → `json_object` → текст); неподдерживаемые режимы отключаются
автоматически после первого 400.

## Входной JSON

См. [examples/demo/input.json](examples/demo/input.json): `case_id`, `brief`,
`repository`, `output_dir`, `author`, `seed`, `limits`, `untrusted_dirs`, `llm`.
Пути в JSON относительны файлу JSON, `--output-dir` из CLI — текущей папке. `output_dir` должен быть пустым и вне репозитория.

Реальный пример на проекте `meridian` (Python + PostgreSQL + Alembic):
[examples/meridian/input.json](examples/meridian/input.json). Сошёлся с первой попытки на
`gpt-oss:120b-cloud`: build ~150 с, Base/Oracle плюс 6 изолированных прогонов ~70 с, 3 вызова LLM.

## Запуск

```bash
# полный прогон с верификацией в Docker
python -m harness run examples/demo/input.json --base-url http://localhost:11434/v1 --model gpt-oss:120b

# то же, но только сборка артефактов без Docker (status = "unverified")
python -m harness run examples/demo/input.json --skip-docker

# оффлайн-демо на mock-ответах LLM (без сервера, без ключей)
python -m harness run examples/demo/input.json --mock-llm examples/demo/mock_responses.json --skip-docker

# только этапы 1-2: паспорт стека + контекстный пакет
python -m harness inspect examples/demo/input.json --no-llm --show-context

# хэш снапшота репозитория
python -m harness snapshot examples/demo_repo
```

Полезные флаги `run`: `--search-backend zvec|keyword|auto`, `--no-isolated`
(пропустить изолированные прогоны по категориям), `--keep-image`.

## Что получается на выходе

```
out/<case_id>/
├── task/
│   ├── task.toml               # schema_version 1.1, case_id, snapshot sha256, 3 списка тестов
│   ├── instruction.md          # ТЗ для решателя (без спойлеров)
│   ├── solution/solve.sh       # эталонное исправление
│   ├── tests/                  # test_*.py, manifest.json, test.sh, verify.py, pytest.ini
│   └── environment/            # Dockerfile + чистая копия репозитория (repo/)
├── evidence/
│   ├── build.log, base/, oracle/, isolated/   # логи прогонов, tests.xml, reward.txt, results.json
│   ├── attempts/NN/            # артефакты неудачных итераций self-healing
│   ├── profile.json, localization.json, synthesis.json, verification.json
│   ├── summary.json            # все запуски: длительность, exit code, статус
│   └── llm_usage.json          # модель, токены, длительность каждого вызова
└── result.json                 # status: ready | failed | unverified, limitations
```

Контракт песочницы: репозиторий в `/app/repo` (рабочая директория), тесты монтируются
в `/tests` (ro), решение в `/solution` (ro), логи в `/logs`. Проверка:
`sh /tests/test.sh [all|fail_to_pass|pass_to_pass|anti_cheat]` → `/logs/verifier/reward.txt`.

## Тесты харнесса

```bash
pytest
```

`tests/test_pipeline_mock.py` прогоняет весь пайплайн на mock-LLM и затем запускает
сгенерированный `verify.py` на хосте: Base (reward 0) и Oracle после `solve.sh`
(reward 1). Для Oracle нужен POSIX `sh` (Git Bash на Windows).

## Ограничения MVP

- Только Python-стек; интерфейсы `IStackDetector` / `IEnvironmentBuilder` / `ITestRunner`
  в [harness/providers/base.py](harness/providers/base.py) — точки расширения.
- PostgreSQL ставится из Debian-репозитория при сборке образа, применяется только
  `alembic upgrade head`; SQL-сиды вручную.
- `PROTOCOL.md` в задании не приложен — структура `task.toml` следует описанию из
  архитектурного документа и может потребовать выравнивания.
- Self-healing перегенерирует весь бандл (тесты + solve.sh) целиком, до `limits.max_retries` раз.
