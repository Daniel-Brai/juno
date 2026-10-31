# juno

Your headless agent harness for long-running tasks and everyday coding. Give it a goal, walk away, and come back to finished work.

juno runs one long agent session against a goal. Use it as a coding agent in a project, or let it work unattended on a task for hours, in a sandbox, with every step saved so a crash never loses work.

## Requirements

- Python 3.15 and [uv](https://docs.astral.sh/uv/)
- An API key: `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` for any model on OpenRouter
- [Podman](https://podman.io/) if you want commands to run in a container (the default for `juno run`)
- `git`, and [ripgrep](https://github.com/BurntSushi/ripgrep) for faster search

## Install

```bash
git clone https://github.com/Daniel-Brai/juno.git
cd juno
uv sync --all-extras
```

The `tui` extra adds the terminal UI and the `mcp` extra adds MCP servers. Run commands with `uv run juno ...`, or install it as a tool with `uv tool install --editable ".[tui,mcp]"`.

## Chat in a project

```bash
cd ~/code/my-project
uv run juno chat
```

This opens the terminal UI. Type a request and press Enter. Type `/` to see commands such as `/model`, `/mode`, `/resume`, `/compact` and `/undo`, and `@path` to attach a file. Edits and commands ask for approval first unless you allow them.

To run one request without the UI:

```bash
uv run juno chat -p "make the failing test pass" --yes
uv run juno chat -p "what does src/app.py do?" --output json
```

Pick up where you left off with `--continue`, or `--resume ID` for an older session. Choose a model with `--provider openrouter --model deepseek/deepseek-v4-flash`, and cap spending with `--max-cost 0.50`.

## Run a task unattended

A task is a folder with `task.toml`, `description.md` and `data/`. Try the example:

```bash
uv run juno run examples/tasks/quickstart --budget 15m --sandbox local
uv run juno status runs/quickstart-...
```

The agent explores, trains, and registers submissions with validation scores; the best one is selected when time runs out. If the run stops, `juno resume RUN` continues it, `juno stop RUN` removes its sandbox, and `juno grade RUN --answers DIR` scores its submissions against hidden answers.

## Experiments

An experiment runs several arms (sets of flags or settings) across tasks and seeds, grades them, and reports the differences with confidence intervals:

```bash
uv run juno eval run examples/experiments/system-tool.toml
uv run juno eval status examples/experiments/system-tool.toml
uv run juno eval report examples/experiments/system-tool.toml
```

`max_cost` and `max_total_cost` in the manifest cap what a run and the whole experiment may spend.

## Configuration

Settings come from `juno.toml` in the current folder, then `JUNO_` environment variables (for example `JUNO_MODEL__EFFORT=low`), then command options. `--set key=value` changes any one setting.

```toml
[model]
provider = "openrouter"
name = "anthropic/claude-opus-5.5"
effort = "medium"

[budget]
total = "6h"
max_cost = 5.0

[sandbox]
kind = "podman"
memory = "16g"
```

In a chat project, juno also reads:

- `AGENTS.md` or `.juno/instructions.md` for project instructions
- `.juno/permissions.toml` for the permission mode and allow or deny rules
- `.juno/mcp.toml` for MCP servers, whose tools appear as `mcp__server__tool`

The last two can also live in `~/.config/juno/` for every project.

## Development

```bash
make setup      # install everything and the git hooks
make check      # lint, types, import contracts and tests
make test-all   # also the slow, GPU and live API tests
```

## License

MIT
