# Guidance for working on pygptreeo

## Comments and docstrings

Keep comments and docstrings short. They should explain the current code:
what it does and, where not obvious, why. Do not describe the development
history (earlier approaches, what was tried and rejected, what a change fixed),
and do not justify a design choice at length; if a longer rationale is worth
keeping, put it in `docs/design.md`, and measurements in `docs/benchmarks.md`,
not in the code. In general, avoid unnecessary jargon.

## README entries

New additions to the main README should be kept short and to the point, keeping 
with the style of what is already there. The README should rarely refer to 
benchmark results or similar.

## Code guidelines

When possible, avoid overly defensive code in the main pygptreeo code. 
And when possible, keep the code structure simple and efficient.
