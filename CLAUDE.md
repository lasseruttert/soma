# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a multi-agent system designed to handle multiple different requests by the user. Additionally it implements a RAG (Retrieval-Augmented Generation) approach to improve the quality of responses and reference user uploaded documents. 


## Basic Commandments
- BE CRITICAL: Never simply agree with the user. Always question and validate their requests to ensure the best outcome.
- ASK QUESTIONS: If you are unsure about something, ask clarifying questions to gather more information. Never assume missing information.
- BE CONCISE: Provide clear and concise responses. Avoid unnecessary verbosity.
- REFACTOR, DONT REWRITE: Improve on the current implementation instead of rewriting it from scratch. If you deem it necessary to make significant changes, discuss them with the user.
- TIDY UP: Keep the code and responses organized and easy to read.
- DOCUMENT: Always update the documentation after every change.
- ONLY DO what is asked.
- DO NOT update CLAUDE.md or GEMINI.md unless explicitly requested by the user.


## Code Style
- NEVER use emojis in the code or comments.
- USE meaningful, descriptive variable, function, and class names.
- USE consistent naming conventions (snake_case for variables and functions, CamelCase for classes).
- USE comments to explain complex logic, but avoid obvious comments.
- USE docstrings for all functions and classes to describe their purpose, parameters, and return values.
- USE type hints for function parameters and return types to improve code readability and maintainability.


## Documentation
- Documentation can be found in the `docs` folder.
- ONLY use markdown for documentation.
- WRITE detailed plans in `docs/plans`, they should be comprehensive and cover all aspects of the implementation. Enough so you can later implement the feature without other context.
- WRITE detailed change logs in `docs/changelog.md` to document all changes made to the codebase.
- WRITE all new plans and change logs in a seperate file. They should be named after the feature. Change logs should have a date in their name.
- ALWAYS update the documentation after every change.


## Conda Environment Policies
- ALWAYS use the conda environment specified in the project: `soma`
- NEVER install or update packages in the base environment.
- UPDATE the requirements.txt file with any new dependencies.
- ALWAYS run code in the conda environment.