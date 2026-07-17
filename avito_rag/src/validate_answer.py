"""Проверка формата answer.csv перед отправкой на платформу."""
import sys, pathlib
import pandas as pd

DATA = pathlib.Path(__file__).parent.parent / "candidate_data"
ans_path = sys.argv[1] if len(sys.argv) > 1 else "answer.csv"

ans = pd.read_csv(ans_path)
test = pd.read_feather(DATA / "test.f")
articles = pd.read_feather(DATA / "articles.f")
valid_ids = set(articles.article_id.tolist())

assert list(ans.columns) == ["query_id", "answer"], f"колонки: {ans.columns.tolist()}"
assert len(ans) == len(test), f"строк {len(ans)}, ожидалось {len(test)}"
assert set(ans.query_id) == set(test.query_id), "query_id не совпадают с test.f"
assert not ans.query_id.duplicated().any(), "дубли query_id"

for _, row in ans.iterrows():
    ids = [int(x) for x in str(row.answer).split()]
    assert 1 <= len(ids) <= 10, f"q{row.query_id}: {len(ids)} статей"
    assert len(ids) == len(set(ids)), f"q{row.query_id}: повторы article_id"
    bad = set(ids) - valid_ids
    assert not bad, f"q{row.query_id}: несуществующие id {bad}"

print(f"OK: {len(ans)} строк, формат корректен")
