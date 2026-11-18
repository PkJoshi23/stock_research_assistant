"""The 4-step schema-grounded text-to-Cypher pipeline.

Step 1  intent detection      graph query vs. live API snapshot fetch
Step 2  schema extraction     choose the relevant subset of the schema
Step 3  cypher translation    generate Cypher from the mini-schema only
Step 4  execution             run against Relate, auto-correct on failure

Each step is a separate module with a typed input/output contract so it can be
unit-tested without hitting the LLM or the database.
"""
