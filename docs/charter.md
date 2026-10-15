# Charter

## Problem
Finding precise answers within a large volume of articles is time-consuming, and answers lacking citations are unreliable.

## User
A researcher who possesses dozens of PDF articles and wishes to ask questions based on them.

## Non-goals
- General-purpose chat without source citations.
- Training a model from scratch.
- Mandatory reliance on paid APIs (local execution via Ollama).

## Success Criteria (Measurable)
- Every answer must include at least one verifiable reference; if no evidence exists, the response must be "I don't know."
- Recall@5 on the golden set must not fall below a predefined threshold (the threshold will be established by Day 10).
- The project must be executable and testable by anyone using a single command.

