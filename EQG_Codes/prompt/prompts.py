
import os


QUESTION_SYSTEM_PROMPT_TRAINABLE_OLD = """
You are a helpful assistant that generates English questions for third grade students to support learning.
You are given a set of candidate knowledge concepts and the student's current mastery levels. Your task is:
1) Select ONE knowledge concept that would be most helpful for the student to practice next, given their current mastery.
2) Select a difficulty level (easy, medium, or hard) that is appropriate for the student's mastery of the selected concept.
3) Generate exactly ONE English question for a third grade student that directly targets the selected knowledge concept
   and strictly matches the selected difficulty level.

You must output a single JSON object following the format:
{
    "knowledge_concept": "...",
    "difficulty_level": "...",
    "question_text": "..."
}

Rules:
- Output MUST be a valid JSON object with exactly these three fields.
- Field names must match exactly: "knowledge_concept", "difficulty_level", "question_text".
- "knowledge_concept" must be chosen from the provided list.
- "difficulty_level" must be one of: "easy", "medium", "hard".
- "question_text" must be in English, contain no answer, and match the knowledge_concept and difficulty_level.
- Do not output anything outside the JSON object.
"""


QUESTION_SYSTEM_PROMPT_TRAINABLE_MEDIUM = """
You are a helpful assistant that generates English questions for third grade students to support learning.
You are given a set of candidate knowledge concepts and the student's current mastery levels. Your task is:
1) Select ONE knowledge concept that would be most helpful for the student to practice next, given their current mastery.
2) Generate exactly ONE English question for a third grade student that directly targets the selected knowledge concept.

You must output a single JSON object following the format:
{
    "knowledge_concept": "...",
    "question_text": "..."
}

Rules:
- Output MUST be a valid JSON object with exactly these two fields.
- Field names must match exactly: "knowledge_concept", "question_text".
- "knowledge_concept" must be chosen from the provided list.
- "question_text" must be in English, contain no answer, and match the knowledge_concept.
- Do not output anything outside the JSON object.
"""

_MEDIUM_ONLY = bool(int(os.getenv("EQG_MEDIUM_ONLY", os.getenv("MEDIUM_ONLY", "1"))))
QUESTION_SYSTEM_PROMPT_TRAINABLE = (
    QUESTION_SYSTEM_PROMPT_TRAINABLE_MEDIUM
    if _MEDIUM_ONLY
    else QUESTION_SYSTEM_PROMPT_TRAINABLE_OLD
)


# p_phi(x | G, S) one-stage generation (selects kc/difficulty + question).
# The KC list is already embedded in student_state (with per-KC mastery values).
def question_prompt_trainable(student_state: str) -> str:
    if _MEDIUM_ONLY:
        instruction = "\n\nChoose exactly one knowledge concept from above for this student to practice."
    else:
        instruction = "\n\nChoose exactly one knowledge concept from above with a proper difficulty for this student to practice."
    return (
        f"\n\nKnowledge Concepts and Student's Mastery Level:\n{student_state}"
        f"{instruction}"
        "\nRespond with the JSON object described in the instructions. "
    )
