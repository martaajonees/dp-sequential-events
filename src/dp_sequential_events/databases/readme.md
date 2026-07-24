# Datasets

This document describes the CSV event logs included in this repository: one development/test file used while building the tool, and three synthetic event logs used for the formal evaluation.

All logs follow the same input format required by the tool:

| Column      | Description                          |
|-------------|---------------------------------------|
| `CaseID`    | Identifier of the case/subject (e.g. student) |
| `Activity`  | Name of the activity/event            |
| `Timestamp` | Date and time at which the event occurred |

---

## 1. `synthetic-data.csv` — Development / test dataset

This file was used during the early development of the differential privacy pipeline to prototype and debug the ϵ-estimation and noise-injection logic before running the full experiments. It is **not** one of the datasets used for the formal evaluation in the thesis — it exists purely as a lightweight sandbox for testing individual pipeline steps (annotation, filtering, ϵ estimation, timestamp anonymization) without the overhead of the larger evaluation logs below.

---

## 2. Evaluation datasets (`synthetic_data_reg1.csv`, `synthetic_data_reg2.csv`, `synthetic_data_reg3.csv`)

These are the three synthetic event logs used to evaluate the proposed method, each simulating a different real-world Learning Analytics (LA) scenario.

### `synthetic_data_reg1.csv` — Academic interaction in virtual courses ("Log 1")

Simulates student interaction with course resources in a virtual subject.

- **Size:** 3,258 students / 15,473 events
- **Activities (8):**
  - `a` — Student enrolls in the course
  - `b` — Visit to the virtual environment homepage
  - `c` — Access to study materials
  - `d` — Participation in discussion forums
  - `e` — Submission of an assignment/exam
  - `f` — Final result: Pass
  - `g` — Final result: Fail
  - `h` — Final result: Dropout

Typical traces show a student enrolling, browsing materials, optionally engaging in forums, and ending in one of three outcomes (pass/fail/dropout).

### `synthetic_data_reg2.csv` — Activity on learning platforms / MOOC environments ("Log 2")

Simulates interaction on a Coursera-like platform for a course with forums, videos and a quiz.

- **Size:** 91 students / 877 events
- **Activities (12):**
  - `a` — View a forum discussion
  - `b` — Access material
  - `c` — Access the forum's main page
  - `d` — Write a forum post
  - `e` — Edit/update a post
  - `f` — Start a quiz attempt
  - `g` — View the quiz summary
  - `h` — Access the quiz main view
  - `i` — Resume a half-finished quiz
  - `j` — View a Moodle content page
  - `k` — Close/submit the quiz
  - `l` — Review quiz results

This is the smallest and structurally most diverse log, which makes it the most sensitive to noise injection in the experiments (see the Experiments README).

### `synthetic_data_reg3.csv` — Eye-tracking log during problem solving ("Log 3")

Simulates eye-tracking data captured while students solve a problem composed of a graph and a text.

- **Size:** 830 students / 4,501 events
- **Activities (7):**
  - `a` — Fixation on the instructions area
  - `b` — Fixation on the main text
  - `c` — Fixation on the graph
  - `d` — Fixation on the question statement
  - `e` — Quick re-fixation on the text
  - `f` — Quick re-fixation on the graph
  - `g` — Click to submit the answer

Students typically follow a structured reading pattern (instructions → text/graph → question), with some looping back to re-check the text or graph before answering.
