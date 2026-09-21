"""Verbatim prompt/template constants for the MIMIC-III EHRAgent target, ported
from EHR/ehragent/prompts_mimic.py (Dong et al., arXiv:2503.03704). Names are
snake/UPPER-cased for our style; string content is byte-for-byte theirs.
"""

from __future__ import annotations

CODE_HEADER = """from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

RETR_KNOWLEDGE = """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
For different tables, they contain the following information:
(1) admissions: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
(2) chartevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(3) cost: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
(4) d_icd_diagnoses: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(5) d_icd_procedures: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(6) d_items: ROW_ID, ITEMID, LABEL, LINKSTO
(7) d_labitems: ROW_ID, ITEMID, LABEL
(8) dianoses_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(9) icustays: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
(10) inputevents_cv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
(11) labevents: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(12) microbiologyevents: RROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
(13) outputevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
(14) patients: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
(15) prescriptions: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
(16) procedures_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(17) transfers: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

{demonstrations}

Question: {question}
Knowledge:
"""

EHRAGENT_MESSAGE_PROMPT = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
Write a python code to solve the given question. You can use the following functions:
(1) Calculate(FORMULA), which calculates the FORMULA and returns the result.
(2) LoadDB(DBNAME) which loads the database DBNAME and returns the database. The DBNAME can be one of the following: admissions, chartevents, cost, d_icd_diagnoses, d_icd_procedures, d_items, d_labitems, diagnoses_icd, icustays, inputevents_cv, labevents, microbiologyevents, outputevents,patients, prescriptions, procedures_icd, transfers.
(3) FilterDB(DATABASE, CONDITIONS), which filters the DATABASE according to the CONDITIONS and returns the filtered database. The CONDITIONS is a string composed of multiple conditions, each of which consists of the column_name, the relation and the value (e.g., COST<10). The CONDITIONS is one single string (e.g., "admissions, SUBJECT_ID=24971").
(4) GetValue(DATABASE, ARGUMENT), which returns a string containing all the values of the column in the DATABASE (if multiple values, separated by ", "). When there is no additional operations on the values, the ARGUMENT is the column_name in demand. If the values need to be returned with certain operations, the ARGUMENT is composed of the column_name and the operation (like COST, sum). Please do not contain " or ' in the argument.
(5) SQLInterpreter(SQL), which interprets the query SQL and returns the result.
(6) Calendar(DURATION), which returns the date after the duration of time.
Use the variable 'answer' to store the answer of the code. Here are some examples:
{examples}
(END OF EXAMPLES)
Knowledge:
{knowledge}
Question: {question}
Solution: """

CODE_DEBUGGER = """Given a question:
{question}
The user have written code with the following functions:
(1) Calculate(FORMULA), which calculates the FORMULA and returns the result.
(2) LoadDB(DBNAME) which loads the database DBNAME and returns the database. The DBNAME can be one of the following: admissions, chartevents, cost, d_icd_diagnoses, d_icd_procedures, d_items, d_labitems, diagnoses_icd, icustays, inputevents_cv, labevents, microbiologyevents, outputevents,patients, prescriptions, procedures_icd, transfers.
(3) FilterDB(DATABASE, CONDITIONS), which filters the DATABASE according to the CONDITIONS. The CONDITIONS is a string composed of multiple conditions, each of which consists of the column_name, the relation and the value (e.g., COST<10). The CONDITIONS is one single string (e.g., "admissions, SUBJECT_ID=24971").
(4) GetValue(DATABASE, ARGUMENT), which returns the values of the column in the DATABASE. When there is no additional operations on the values, the ARGUMENT is the column_name in demand. If the values need to be returned with certain operations, the ARGUMENT is composed of the column_name and the operation (like COST, sum). Please do not contain " or ' in the argument.
(5) SQLInterpreter(SQL), which interprets the query SQL and returns the result.
(6) Calendar(DURATION), which returns the date after the duration of time.

The code is as follows:
{code}

The execution result is:
{error_info}

Please check the code and point out the most possible reason to the error.
"""

# Real bug/quirk in their shipped medagent.py's error_debugger(): the "call an
# LLM to diagnose the error" path is dead code -- `prediction` is hardcoded to
# this literal string, never actually generated by a model, and the try block
# can't raise (there's no LLM call in it), so this string is what "Potential
# Reasons: ..." always says in their real logs. CODE_DEBUGGER above (the real
# prompt they'd send an LLM) is therefore unused in their actual execution
# path -- kept here anyway since it's their real prompt text, and swap_temp of
# a real call is one plausible fidelity knob (see agent.py).
STUB_DEBUG_REASON = "Debugging response here"

SYSTEM_PROMPT = """You are a helpful AI assistant. Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh
coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or
search the web, download/read a file, print the content of a webpage or a file, get the current
date/time. After sufficient info is printed and the task is ready to be solved based on your
language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the
result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be
clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any
other feedback or perform any other action beyond executing the code you suggest. The user can't
modify your code. So do not suggest incomplete code which requires users to modify. Don't use a
code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename>
inside the code block as the first line. Don't include multiple code blocks in one response. Do not
ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the
full code instead of partial code or code changes. If the error can't be fixed or if the task is
not solved even after the code is executed successfully, analyze the problem, revisit your
assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response
if possible.
Reply "TERMINATE" in the end when everything is done."""

# The real system_message actually passed to autogen.agentchat.AssistantAgent
# in main.py -- shorter than SYSTEM_PROMPT above (which is toolset_high.py's
# unused llm_agent() variant; main.py's chatbot never uses it).
CHATBOT_SYSTEM_MESSAGE = (
    "For coding tasks, only use the functions you have been provided with. "
    "Reply TERMINATE when the task is done. Save the answers to the questions "
    "in the variable 'answer'. Please only generate the code."
)

RETR_KNOWLEDGE_SYSTEM_MESSAGE = "You are an AI assistant that helps people find information."

CODE_DEBUGGER_SYSTEM_MESSAGE = (
    "You are an AI assistant that helps people debug their code. Only list "
    "one most possible reason to the errors."
)

# Real 4-shot memory seed, ported verbatim from EHRAgent_4Shots_Knowledge in
# prompts_mimic.py -- main.py parses this into the initial long_term_memory
# when no --load_memory_path is given (see data.py's parse_seed_memory()).
EHRAGENT_4SHOTS_KNOWLEDGE = """Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge:
- As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
- The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
- The HADM_ID can be used to find the corresponding COST in the cost database.
Solution: date = Calendar('-1 year')
# As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
diagnosis_db = LoadDB('d_icd_diagnoses')
filtered_diagnosis_db = FilterDB(diagnosis_db, 'SHORT_TITLE=comp-oth vasc dev/graft')
icd_code = GetValue(filtered_diagnosis_db, 'ICD9_CODE')
# The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
diagnoses_icd_db = LoadDB('diagnoses_icd')
filtered_diagnoses_icd_db = FilterDB(diagnoses_icd_db, 'ICD9_CODE={}'.format(icd_code))
hadm_id_list = GetValue(filtered_diagnoses_icd_db, 'HADM_ID, list')
# The HADM_ID can be used to find the corresponding COST in the cost database.
max_cost = 0
for hadm_id in hadm_id_list:
    cost_db = LoadDB('cost')
    filtered_cost_db = FilterDB(cost_db, 'HADM_ID={}'.format(hadm_id))
    cost = GetValue(filtered_cost_db, 'COST, sum')
    if cost > max_cost:
        max_cost = cost
answer = max_cost

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge:
- We can find the visiting information of patient 2238 in the admissions database.
- As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
- As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
- We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit.
Solution: # We can find the visiting information of patient 2238 in the admissions database.
patient_db = LoadDB('admissions')
filtered_patient_db = FilterDB(patient_db, 'SUBJECT_ID=2238||min(DISCHTIME)')
hadm_id = GetValue(filtered_patient_db, 'HADM_ID')
# As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
d_items_db = LoadDB('d_items')
filtered_d_items_db = FilterDB(d_items_db, 'LABEL=tpn w/lipids')
item_id = GetValue(filtered_d_items_db, 'ITEMID')
# As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
icustays_db = LoadDB('icustays')
filtered_icustays_db = FilterDB(icustays_db, 'HADM_ID={}'.format(hadm_id))
icustay_id = GetValue(filtered_icustays_db, 'ICUSTAY_ID')
# We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit.
inputevents_cv_db = LoadDB('inputevents_cv')
filtered_inputevents_cv_db = FilterDB(inputevents_cv_db, 'HADM_ID={}||ICUSTAY_ID={}||ITEMID={}'.format(hadm_id, icustay_id, item_id))
if len(filtered_inputevents_cv_db) > 0:
    answer = 1
else:
    answer = 0

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge:
- We can find the visiting information of patient 58730 in the admissions database.
- As procedures are stored in the procedures_icd database, we can find the corresponding ICD9_CODE in the procedures_icd database.
- As we only need to find the name of the procedure, we can find the corresponding SHORT_TITLE as the name in the d_icd_procedures database.
Solution: answer = SQLInterpreter('select d_icd_procedures.short_title from d_icd_procedures where d_icd_procedures.icd9_code in ( select t1.icd9_code from ( select procedures_icd.icd9_code, count( procedures_icd.charttime ) as c1 from procedures_icd where procedures_icd.hadm_id in ( select admissions.hadm_id from admissions where admissions.subject_id = 58730 ) group by procedures_icd.icd9_code ) as t1 where t1.c1 >= 2 )')

Question: calculate the length of stay of the first stay of patient 27392 in the icu.
Knowledge:
- We can find the visiting information of patient 27392 in the admissions database.
- As we only need to find the length of stay, we can find the corresponding INTIME and OUTTIME in the icustays database.
Solution: from datetime import datetime
patient_db = LoadDB('admissions')
filtered_patient_db = FilterDB(patient_db, 'SUBJECT_ID=27392||min(ADMITTIME)')
hadm_id = GetValue(filtered_patient_db, 'HADM_ID')
icustays_db = LoadDB('icustays')
filtered_icustays_db = FilterDB(icustays_db, 'HADM_ID={}'.format(hadm_id))
intime = GetValue(filtered_icustays_db, 'INTIME')
outtime = GetValue(filtered_icustays_db, 'OUTTIME')
intime = datetime.strptime(intime, '%Y-%m-%d %H:%M:%S')
outtime = datetime.strptime(outtime, '%Y-%m-%d %H:%M:%S')
length_of_stay = outtime - intime
if length_of_stay.seconds // 3600 > 12:
    answer = length_of_stay.days + 1
else:
    answer = length_of_stay.days
"""


def parse_seed_memory(raw: str = EHRAGENT_4SHOTS_KNOWLEDGE) -> list[dict]:
    """Matches main.py's inline parsing of EHRAgent_4Shots_Knowledge into the
    initial long_term_memory list of {"question","knowledge","code"} dicts:
    split on blank lines, then split each block on 'Question:' / '\\nKnowledge:\\n'
    / '\\nSolution:' -- same literal string surgery as their code."""
    items = []
    for item in raw.split("\n\n"):
        item = item.split("Question:")[-1]
        question = item.split("\nKnowledge:\n")[0]
        item = item.split("\nKnowledge:\n")[-1]
        knowledge = item.split("\nSolution:")[0]
        code = item.split("\nSolution:")[-1]
        items.append({"question": question, "knowledge": knowledge, "code": code})
    return items
