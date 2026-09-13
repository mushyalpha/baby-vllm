import os

with open("day4/DAY4_RESULTS.md", "r") as f:
    day4 = f.read()

with open("day5/DAY5_NARRATIVE.md", "r") as f:
    day5_narrative = f.read()

with open("day5/DAY5_RESULTS.md", "r") as f:
    day5_results = f.read()

with open("CODE.md", "r") as f:
    code_content = f.read()

header = f"""# Baby-vLLM Project Context (Days 1 - 5)

## Day 4: Roofline Profiling
{day4}

---

## Day 5 Narrative
{day5_narrative}

---

## Day 5 Results Summary
{day5_results}

---
# Codebase

"""

new_content = header + code_content

with open("CODE.md", "w") as f:
    f.write(new_content)
