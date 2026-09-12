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

## Входной JSON (protocol 1.0)

Эталон формата: [examples/protocol/input.example.json](examples/protocol/input.example.json).

| Поле | Назначение |
|------|------------|
| `protocol_version` | `"1.0"` |
| `repository`, `output_dir` | пути относительно файла JSON; `output_dir` пустой и вне репозитория |
| `brief` | постановка задачи (обязательно непустая) |
| `case_id`, `source`, `team` | идентификаторы, `case_id` может содержать `/` (`hackathon/settlement-001`) |
| `difficulty` | `easy` / `medium` / `hard`, передаётся в промпт и в `task.toml` |
| `language` | язык `instruction.md` (`ru`, `en`, …), не язык стека |
| `limits` | `agent_timeout_sec` (бюджет решателя, в `task.toml`), `verifier_timeout_sec` (один прогон `test.sh`), `build_timeout_sec`, `cpus` и `memory_mb` (ограничения `docker run`), `storage_mb` |
| `author` | `{ "name", "email" }` |
| `seed` | целое |
| `untrusted_dirs`, `llm`, `limits.max_retries`, `limits.max_context_*` | необязательные расширения харнесса |

`--output-dir` из CLI резолвится относительно текущей папки. Старые формы (`author` строкой,
`limits.run_timeout_sec`) принимаются.

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

## Разбор брифа (этап 2a)

Бриф может быть любым: одна фраза, тикет, спецификация со списком, вставленная переписка,
на русском или английском, либо путь к `.md`/`.txt` файлу. Первым LLM-вызовом харнесс
превращает его в спецификацию (`evidence/brief_spec.json`):

- `requirements` R1…Rn: атомарные проверяемые требования с критериями приёмки и типом
  (`bug` / `feature` / `change` / `invariant` / `performance` / `refactor`);
- `constraints` (что нельзя менять → anti_cheat), `out_of_scope`, `entities`, `search_queries`,
  `candidate_files`;
- `assumptions` и `ambiguities`: при неоднозначности харнесс не останавливается, а фиксирует
  принятую трактовку и открытые вопросы (они попадают в `limitations` в `result.json`).

Дальше спецификация управляет поиском кода и синтезом: каждое требование `bug/feature/change`
обязано покрываться хотя бы одним `fail_to_pass`-тестом, `invariant` — `pass_to_pass`,
ограничения — `anti_cheat`. Модель возвращает карту `coverage`, валидатор отклоняет бандл с
непокрытым требованием и отправляет на раунд repair. Без LLM (`inspect --no-llm`) работает
эвристический разбор: весь бриф как одно требование.

### Защиты в контуре self-healing

Модели ошибаются предсказуемо, поэтому часть исправлений харнесс делает сам, без LLM:

- тест из `fail_to_pass`, который проходит и на Base, и на Oracle, переносится в `pass_to_pass`,
  а его требование понижается до `invariant` (перепроверка без вызова модели);
- `solve.sh` в раунде лечения разрешено менять только при бизнес-падении `fail_to_pass` на Oracle
  или крахе самого `solve.sh`; «подгонка» решения под неверный тест отклоняется;
- `async def test_` без `pytest-asyncio`, строковые сравнения сигнатур и прочие типовые
  ошибки ловятся валидатором до Docker и уходят в repair-раунд;
- невалидный результат лечения не перепроверяется в песочнице, а сразу лечится заново
  с вердиктом валидатора; падение `docker build` из-за сети повторяется один раз.

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
