{# version: 1 #}
You are labeling sentences from research paper abstracts.

Task: assign the sentence exactly one label from: background, method, result, limitation.

Rules:
- Use only these four labels, in lowercase.
- Think step by step in at most three short sentences.
- Then finish with a final line in the form: Answer: <label>

Sentence: """{{ sentence }}"""

