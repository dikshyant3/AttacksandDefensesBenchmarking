"""Real data loading for the MINJA EHRAgent/MIMIC-III target (Dong et al.,
arXiv:2503.03704). All files vendored directly from EHR/ehragent/ehrsql-ehragent/
mimic_iii/ in their repo: the real MIMIC-III demo CSV tables (17 tables, the
same PhysioNet demo subset their repo itself ships) and 581 real EHRSQL
questions with ground-truth answers (valid_preprocessed.json).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark.attacks.minja_ehr.tools import _TABLE_FILES, DATA_DIR


@dataclass(frozen=True)
class EHRQuestion:
    qid: str
    question: str
    template: str
    answer: tuple[str, ...]
    department: str


def load_raw_records(path: Path | None = None) -> list[dict]:
    """The real, unparsed valid_preprocessed.json records (dicts with
    db_id/question/template/query/value/q_tag/t_tag/o_tag/tag/department/
    importance/para_type/is_impossible/split/id/answer keys) -- needed as-is
    (not the narrowed EHRQuestion view) by poison.make_poison_templates(),
    which rewrites several of these raw fields directly."""
    path = path or (DATA_DIR / "valid_preprocessed.json")
    return json.loads(path.read_text())


def load_questions(path: Path | None = None) -> tuple[EHRQuestion, ...]:
    """The real 581-question valid_preprocessed.json set -- natural-language
    question, a template variant (what add_poison.py appends its redirect
    sentence to), and ground-truth answer(s) for scoring."""
    path = path or (DATA_DIR / "valid_preprocessed.json")
    records = json.loads(path.read_text())
    questions = []
    for r in records:
        answer = r.get("answer") or []
        if isinstance(answer, str):
            answer = [answer]
        questions.append(
            EHRQuestion(
                qid=r["id"],
                question=r["question"],
                template=r["template"],
                answer=tuple(str(a) for a in answer),
                department=r.get("department", ""),
            )
        )
    return tuple(questions)


def questions_mentioning_patient(questions: tuple[EHRQuestion, ...], patient_id: int) -> tuple[EHRQuestion, ...]:
    """Real questions whose SQL-independent template/question text references
    a specific patient's SUBJECT_ID -- these are what a redirect attack for
    that patient would actually target. Matches on the literal numeral
    appearing in the question text (how these real EHRSQL questions phrase a
    specific patient, e.g. "...for patient 30789...")."""
    needle = str(patient_id)
    return tuple(q for q in questions if needle in q.question)


_SQLITE_PATH = DATA_DIR / "mimic_iii.db"


def build_sqlite_db(force: bool = False) -> Path:
    """Builds a local SQLite database from the vendored CSVs, for
    SQLInterpreter()/Calendar() -- matches their own mimic_iii.db, just built
    fresh from the same real CSV content instead of shipping a binary. Cached
    on disk after the first build (the largest tables, e.g. COST.csv at
    ~29MB, make rebuilding on every run wasteful)."""
    if _SQLITE_PATH.exists() and not force:
        return _SQLITE_PATH
    if _SQLITE_PATH.exists():
        _SQLITE_PATH.unlink()

    con = sqlite3.connect(_SQLITE_PATH)
    try:
        for table_name, filename in _TABLE_FILES.items():
            frame = pd.read_csv(DATA_DIR / filename)
            frame.to_sql(table_name, con, index=False)
    finally:
        con.close()
    return _SQLITE_PATH
