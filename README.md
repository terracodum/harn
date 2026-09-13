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

# то же, но только сборка артефактов без Docker (result.json: status = "failed", в limitations — "not verified")
python -m harness run examples/demo/input.json --skip-docker

# оффлайн-демо на mock-ответах LLM (без сервера, без ключей)
python -m harness run examples/demo/input.json --mock-llm examples/demo/mock_responses.json --skip-docker

# только этапы 1-2: паспорт стека + контекстный пакет
python -m harness inspect examples/demo/input.json --no-llm --show-context

# хэш снапшота репозитория
python -m harness snapshot examples/demo_repo
```

Полезные флаги `run`: `--search-backend zvec|keyword` (по умолчанию `zvec`; `keyword` — осознанный
выбор, а не автоматическая замена при отсутствии zvec), `--no-isolated` (пропустить изолированные
прогоны по категориям), `--keep-image`. В `limits` входного JSON можно выключить отсечение
несошедшихся требований: `"prune_unconverged": false`.

## Никакой тихой деградации LLM-шагов

Правило проекта, закреплённое тестом `tests/test_no_silent_fallbacks.py`:

1. Любое исключение из вызова модели (сеть, обрезка по `max_tokens`, невалидный JSON, отсутствующие
   ключи, невалидное содержимое ответа) доходит до оркестратора как `LLMError` / `SynthesisError` /
   `PipelineError` и попадает в `result.json`: `status: "failed"`, `failed_stage`, текст ошибки.
2. Эвристики (`inspect --no-llm`, `--search-backend keyword`) — режимы, выбранные пользователем явно.
   Ни одна из них не включается из `except`.
3. Единственный допустимый ответ на сбой модели — ограниченный и залогированный повтор того же шага
   (разбор брифа: две попытки; repair/heal: `limits.max_retries`), после которого шаг падает.
4. Ответ модели не «подправляется» молча: неизвестный `kind`, пустое требование, лишний или
   недостающий шаг решения — это отклонение бандла с перечнем проблем, а не коррекция по умолчанию.
5. Смена `response_format` в OpenAI-клиенте происходит только на HTTP 400, в тексте которого сервер
   называет `response_format`/схему; любой другой 400 — ошибка, а не повод переслать запрос иначе.
   Использованный режим пишется в `evidence/llm_usage.json` (`mode`).

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
обязано покрываться хотя бы одним `fail_to_pass`-тестом и получает ровно один шаг решения
`solve_R<i>.sh`; `invariant` покрывается только `pass_to_pass` и шага не имеет; ограничения —
`anti_cheat`. Модель возвращает карту `coverage`, валидатор отклоняет бандл с непокрытым
требованием, лишним или недостающим шагом и отправляет на раунд repair. Типы требований после
разбора не меняются: если тест `fail_to_pass` прошёл на исходном коде, это дефект теста, который
уходит в лечение, а не повод перевести требование в `invariant`. Сбой разбора брифа моделью
останавливает прогон; эвристический разбор (весь бриф как одно требование) существует только в
явном режиме `inspect --no-llm`.

## Цепочка решения `solve_R<i>.sh`

Модель не пишет shell: каждый шаг решения — список структурных правок
`{"op": "replace"|"create"|"delete", "path", "old", "new", "content"}`, ровно один шаг на требование
типа `bug/feature/change`. Из них харнесс сам:

- применяет правки R1..Ri-1 в памяти и показывает модели в раунде лечения файлы в состоянии перед
  шагом, который надо чинить (`<repository_state after="R1,R2">`);
- отклоняет бандл до Docker, если `old` не встречается ровно один раз в состоянии после предыдущих
  шагов, файл для `create` уже существует, путь ведёт за пределы репозитория или в `tests/`;
- знает, какие поздние шаги перестанут применяться при отсечении раннего, и отсекает их вместе;
- рендерит скрипты детерминированно:

```
task/solution/
├── solve.sh        # самодостаточный: все шаги инлайном, перед каждым печатает "[solve] step R<i>"
├── solve_R1.sh     # тот же шаг R1 отдельно (доп. файл решения, нужен ступенчатой диагностике)
└── solve_R3.sh     # шаг R3 отдельно (R2 — инвариант, шага нет)
```

Зачем цепочка: при лечении R3 модель не может «забыть» R1 и R2. Требования, которые уже сошлись (все их
тесты ведут себя правильно на Base и Oracle), замораживаются: их шаги передаются в промпт как
`<frozen>`, а бандл, который их изменил или перекатегоризировал их тесты, отклоняется до песочницы.
Проблемы группируются по требованиям (`<focus>`), а при падении Oracle на цепочке из нескольких
шагов харнесс прогоняет префиксы R1, R1+R2, … и сообщает модели, с какого шага начинается сбой
(`evidence/verification.json`, поле `staged`). Если `solve.sh` завершился с ошибкой, в проблему
попадает имя упавшего шага (`solve.sh exited with 1 in step R3`).

### Защиты в контуре self-healing

Харнесс ничего не переклассифицирует сам: вердикт песочницы передаётся модели как есть.

- шаги решения в раунде лечения разрешено менять только при бизнес-падении `fail_to_pass` на
  Oracle или крахе самого шага; «подгонка» решения под неверный тест отклоняется;
- шаги и тесты сошедшихся требований заморожены (см. выше), лечение работает только по `<focus>`;
- `async def test_` без `pytest-asyncio`, строковые сравнения сигнатур и прочие типовые
  ошибки ловятся валидатором до Docker и уходят в repair-раунд;
- невалидный результат лечения не перепроверяется в песочнице, а сразу лечится заново
  с вердиктом валидатора; падение `docker build` из-за сети повторяется один раз.

### Отсечение несошедшихся требований (pruning)

Когда попытки лечения исчерпаны, а часть требований так и не сошлась, харнесс не роняет кейс
целиком: если каждая оставшаяся проблема привязана к конкретному требованию (через его тесты
или упавший шаг), эти требования отсекаются — шаг `solve_R<i>.sh`, их тесты и запись `coverage`
удаляются, `instruction.md` переписывается моделью без них (отдельный вызов
`rewrite_instruction`, проверяется на спойлеры; его сбой роняет кейс), и бандл верифицируется
ещё раз. Всё фиксируется явно: `evidence/pruned.json`, `pruned` в `brief_spec.json`,
`limitations` в `result.json`. Отсечение запрещено, если после него не останется ни одного
`bug/feature/change` требования или ни одного `fail_to_pass` теста, либо если есть проблемы,
которые нельзя привязать к требованию (изолированные прогоны, таймауты, падающие `anti_cheat`):
тогда `status: "failed"` с пояснением.

## Что получается на выходе

```
out/<case_id>/
├── task/
│   ├── task.toml               # PROTOCOL.md §4: [task], [metadata] с тремя списками тестов, [agent], [verifier], [environment]
│   ├── instruction.md          # ТЗ для решателя (без спойлеров)
│   ├── solution/               # solve.sh (оркестратор) + solve_R<i>.sh по требованиям
│   ├── tests/                  # test_*.py, manifest.json, test.sh, verify.py, pytest.ini
│   └── environment/            # Dockerfile + чистая копия репозитория (repo/)
├── evidence/
│   ├── build.log, base/, oracle/, isolated/   # логи прогонов, tests.xml, reward.txt, results.json
│   ├── attempts/NN/            # артефакты неудачных итераций self-healing (+ staged/ префиксные прогоны)
│   ├── profile.json, localization.json, synthesis.json, verification.json, pruned.json
│   ├── summary.json            # все запуски: длительность, exit code, статус
│   └── llm_usage.json          # модель, токены, длительность каждого вызова
└── result.json                 # PROTOCOL.md §2: status ready | failed, task_path, evidence_path, limitations, input_snapshot_sha256
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
- Self-healing перегенерирует весь бандл (тесты + шаги решения) целиком, до `limits.max_retries` раз;
  замораживание сошедшихся требований, `<focus>` и `<repository_state>` (файлы после предыдущих
  шагов) сужают задачу модели, но отдельного LLM-вызова на каждый шаг пока нет.
- Шаги решения выражаются только правками `replace/create/delete`; произвольный shell в эталонном
  решении не поддерживается (переименования, бинарные файлы, генерация кода командами).
