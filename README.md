# homeroom-ml
Repo housing automated tutor for Homeroom web app. Hint content is generated with custom algorithms, and code from https://github.com/patil-suraj/question_generation is involved in phrasing the hint as a question; code from this repo is in directory `\question_generation`. Huge props to them!

## Install

- Download and install Anaconda
- Create a virtual environment, activate it
- Run `pip install -r requirements.txt`

If this fails, try to install pytorch as described here first: https://pytorch.org/get-started/locally/.

After installing all dependencies, run `python setup.py`.

## Usage

To start server:
- Run `python main.py`

Specify the solution of problem currently being worked on:
- Have client send a request to `/load_soln`. Include argument `soln` in request specifying full solution.

Get a hint:
- Send request to `/get_hint`. Include argument `state` in request specifying user's current progress in natural language.

Make sure to specify a solution with `/load_soln` before requesting a hint.
