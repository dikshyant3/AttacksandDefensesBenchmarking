"""Real EHRAgent tool functions, ported from EHR/ehragent/tools/tabtools.py and
tools/calculator.py (Dong et al., "Memory Injection Attacks on LLM Agents via
Query-Only Interaction", arXiv:2503.03704, MIMIC-III/EHRAgent target).

These are the exact functions the agent's generated code calls (via CodeHeader's
name-binding: LoadDB=db_loader, FilterDB=data_filter, GetValue=get_value,
SQLInterpreter=sql_interpreter, Calendar=date_calculator, Calculate=calculator).
Ported faithfully -- same string-based filter DSL, same aggregate operations,
same Levenshtein-based "did you mean" error message -- with paths pointed at
our vendored copies of the real MIMIC-III demo CSVs instead of theirs.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

_TABLE_FILES = {
    "admissions": "ADMISSIONS.csv",
    "chartevents": "CHARTEVENTS.csv",
    "cost": "COST.csv",
    "d_icd_diagnoses": "D_ICD_DIAGNOSES.csv",
    "d_icd_procedures": "D_ICD_PROCEDURES.csv",
    "d_items": "D_ITEMS.csv",
    "d_labitems": "D_LABITEMS.csv",
    "diagnoses_icd": "DIAGNOSES_ICD.csv",
    "icustays": "ICUSTAYS.csv",
    "inputevents_cv": "INPUTEVENTS_CV.csv",
    "labevents": "LABEVENTS.csv",
    "microbiologyevents": "MICROBIOLOGYEVENTS.csv",
    "outputevents": "OUTPUTEVENTS.csv",
    "patients": "PATIENTS.csv",
    "prescriptions": "PRESCRIPTIONS.csv",
    "procedures_icd": "PROCEDURES_ICD.csv",
    "transfers": "TRANSFERS.csv",
}

_table_cache: dict[str, pd.DataFrame] = {}


def db_loader(target_ehr: str) -> pd.DataFrame:
    """LoadDB(DBNAME) -- matches their db_loader exactly, aside from pointing at
    our vendored CSVs. Cached in-process (their version re-reads the CSV from
    disk every call; identical CSV content either way, we just avoid re-parsing
    the larger tables -- e.g. COST.csv, ~29MB -- on every single LoadDB call
    within one agent run)."""
    if target_ehr not in _table_cache:
        _table_cache[target_ehr] = pd.read_csv(DATA_DIR / _TABLE_FILES[target_ehr])
    return _table_cache[target_ehr]


def _levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a):
        current_row = [i + 1]
        for j, char_b in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char_a != char_b)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def data_filter(data: pd.DataFrame, argument: str) -> pd.DataFrame:
    """FilterDB(DATABASE, CONDITIONS) -- ported line-for-line from their
    data_filter: a small string DSL ('||'-joined conditions, each one
    column<op>value, ops in order >=, <=, >, <, =, ' in ', max(...), min(...)),
    same operator precedence via if/elif chaining, same type-coercion attempt
    against the column's existing dtype, same Levenshtein-based error hint when
    an exact-match filter yields zero rows."""
    backup_data = data
    commands = argument.split("||")
    column_name = ""
    value = ""
    for command_str in commands:
        try:
            if ">=" in command_str:
                command = command_str.split(">=")
                column_name, value = command[0], command[1]
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] >= value]
            elif "<=" in command_str:
                command = command_str.split("<=")
                column_name, value = command[0], command[1]
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] <= value]
            elif ">" in command_str:
                command = command_str.split(">")
                column_name, value = command[0], command[1]
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] > value]
            elif "<" in command_str:
                command = command_str.split("<")
                column_name, value = command[0], command[1]
                if value and value[0] in "'\"":
                    value = value[1:-1]
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] < value]
            elif "=" in command_str:
                command = command_str.split("=")
                column_name, value = command[0], command[1]
                if value and value[0] in "'\"":
                    value = value[1:-1]
                try:
                    exemplar = backup_data[column_name].tolist()[0]
                    value = type(exemplar)(value)
                except Exception:
                    pass
                data = data[data[column_name] == value]
            elif " in " in command_str:
                command = command_str.split(" in ")
                column_name, value = command[0], command[1]
                value_list = [s.strip() for s in value.strip("[]").split(",")]
                value_list = [s.strip("'").strip('"') for s in value_list]
                value_list = list(map(type(data[column_name].iloc[0]), value_list))
                data = data[data[column_name].isin(value_list)]
            elif "max" in command_str:
                column_name = command_str.split("max(")[1].split(")")[0]
                data = data[data[column_name] == data[column_name].max()]
            elif "min" in command_str:
                column_name = command_str.split("min(")[1].split(")")[0]
                data = data[data[column_name] == data[column_name].min()]
        except Exception:
            if column_name not in data.columns.tolist():
                columns = ", ".join(data.columns.tolist())
                raise Exception(
                    f"The filtering query {command_str} is incorrect. Please modify the column "
                    f"name or use LoadDB to read another table. The column names in the current "
                    f"DB are {columns}."
                )
            if column_name == "" or value == "":
                raise Exception(
                    f"The filtering query {command_str} is incorrect. There is syntax error in "
                    f"the command. Please modify the condition or use LoadDB to read another table."
                )
        if len(data) == 0:
            column_values = list(set(backup_data[column_name].tolist()))
            if (
                ("=" in command_str)
                and (value not in column_values)
                and (">=" not in command_str)
                and ("<=" not in command_str)
            ):
                distances = {cv: _levenshtein_distance(str(cv), str(value)) for cv in column_values}
                closest = sorted(distances.items(), key=lambda x: x[1])[:5]
                closest_str = ", ".join(str(cv) for cv, _ in closest)
                raise Exception(
                    f"The filtering query {command_str} is incorrect. There is no {value} value "
                    f"in the column. Five example values in the column are {closest_str}. Please "
                    f"check if you get the correct {column_name} value."
                )
            return data
    return data


def get_value(data: pd.DataFrame, argument: str):
    """GetValue(DATABASE, ARGUMENT) -- ported from their get_value: with a bare
    column name, returns the single value (or a comma-joined set of distinct
    values if the filtered frame has more than one row); with 'column, op',
    applies mean/max/min/sum/list."""
    try:
        commands = argument.split(", ")
        if len(commands) == 1:
            column = argument.strip("[]'")
            if len(data) == 1:
                return str(data.iloc[0][column])
            answer_list = sorted({str(v) for v in data[column].tolist()})
            return ", ".join(answer_list)

        column = commands[0]
        op = commands[-1]
        if "mean" in op:
            values = [float(v) for v in data[column].tolist()]
            return sum(values) / len(values)
        if "max" in op:
            try:
                values = [float(v) for v in data[column].tolist()]
            except ValueError:
                values = [str(v) for v in data[column].tolist()]
            return max(values)
        if "min" in op:
            try:
                values = [float(v) for v in data[column].tolist()]
            except ValueError:
                values = [str(v) for v in data[column].tolist()]
            return min(values)
        if "sum" in op:
            values = [float(v) for v in data[column].tolist()]
            return sum(values)
        if "list" in op:
            return [str(v) for v in data[column].tolist()]
        raise Exception(f"The operation {op} contains syntax errors. Please check the arguments.")
    except Exception:
        column_values = ", ".join(data.columns.tolist())
        raise Exception(
            f"The column name is incorrect. Please check the column name and make necessary "
            f"changes. The columns in this table include {column_values}."
        )


_SQLITE_PATH = DATA_DIR / "mimic_iii.db"


def sql_interpreter(command: str):
    """SQLInterpreter(SQL) -- raw SQLite query against a local DB built from the
    same vendored CSVs (see build_sqlite_db() in data.py)."""
    con = sqlite3.connect(_SQLITE_PATH)
    try:
        cur = con.cursor()
        return cur.execute(command).fetchall()
    finally:
        con.close()


def date_calculator(argument: str):
    """Calendar(DURATION) -- SQLite datetime arithmetic relative to "now"."""
    try:
        con = sqlite3.connect(_SQLITE_PATH)
        try:
            cur = con.cursor()
            command = f"select datetime(current_time, '{argument}')"
            return cur.execute(command).fetchall()[0][0]
        finally:
            con.close()
    except Exception:
        raise Exception(
            f"The date calculator {argument} is incorrect. Please check the syntax and make "
            f"necessary changes. For the current date and time, please call Calendar('0 year')."
        )


def calculator(query: str) -> float:
    """Calculate(FORMULA) -- their real calculator() (the WolframAlpha path
    needs a paid API key they never actually provide a real default for, so it
    always raises in their own code too unless someone supplies their own key;
    reproducing the always-available arithmetic path is what's actually load-bearing)."""
    from operator import add, mul, pow as _pow, sub, truediv  # noqa: F401 (pow kept for parity)

    operators = {"+": add, "-": sub, "*": mul, "/": truediv}
    query = re.sub(r"\s+", "", query)
    if query.replace(".", "", 1).isdigit():
        return float(query)
    for symbol, op in operators.items():
        left, sep, right = query.partition(symbol)
        if sep:
            return round(op(calculator(left), calculator(right)), 2)
    raise Exception("Invalid input query for Calculate.")
