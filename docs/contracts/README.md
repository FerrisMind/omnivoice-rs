# Behavior Contracts

These documents are the low-level behavior contracts for the OmniVoice Rust port.

They are not introductory docs. Use them when you need exact implementation expectations for a phase of the pipeline.

## Available Phases

### Phase 8

- [frontend](./phase8/frontend.md)
- [reference-prompt](./phase8/reference-prompt.md)
- [stage0](./phase8/stage0.md)
- [stage1-postprocess](./phase8/stage1-postprocess.md)

### Phase 9

- [frontend](./phase9/frontend.md)
- [reference-prompt](./phase9/reference-prompt.md)
- [stage0](./phase9/stage0.md)
- [stage1-postprocess](./phase9/stage1-postprocess.md)

### Phase 10

- [frontend](./phase10/frontend.md)
- [reference-prompt](./phase10/reference-prompt.md)
- [stage0](./phase10/stage0.md)
- [stage1-postprocess](./phase10/stage1-postprocess.md)

## When to Use These

Use the contracts docs for:

- parity work against upstream artifacts
- debugging behavior regressions
- verifying frontend and stage-level assumptions
- understanding what changed between phases

Use the higher-level docs in [`docs/`](../README.md) for:

- CLI usage
- server usage
- generation controls
- voice design
- language selection
- evaluation workflow
