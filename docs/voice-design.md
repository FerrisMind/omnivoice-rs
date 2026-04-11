# Voice Design

Voice Design mode generates speech from text plus a speaker description string in `instruct`.

No reference audio is needed.

## Quick CLI Example

```powershell
cargo run -p omnivoice-cli -- infer `
  --model model `
  --text "This is a voice design example." `
  --language en `
  --instruct "female, low pitch, british accent" `
  --output out\design.wav `
  --device cpu `
  --dtype f32
```

## How It Works in This Port

The Rust port normalizes `instruct` items before prompt construction.

Supported behavior:

- accepts English and Chinese attributes
- accepts `,` and `，`
- normalizes case
- rejects unsupported values
- suggests likely matches for typos
- rejects conflicting items inside the same category
- rejects mixing Chinese dialect and English accent in one instruction

## Supported Attributes

### Gender

| English | Chinese |
|---|---|
| `male` | `男` |
| `female` | `女` |

### Age

| English | Chinese |
|---|---|
| `child` | `儿童` |
| `teenager` | `少年` |
| `young adult` | `青年` |
| `middle-aged` | `中年` |
| `elderly` | `老年` |

### Pitch

| English | Chinese |
|---|---|
| `very low pitch` | `极低音调` |
| `low pitch` | `低音调` |
| `moderate pitch` | `中音调` |
| `high pitch` | `高音调` |
| `very high pitch` | `极高音调` |

### Style

| English | Chinese |
|---|---|
| `whisper` | `耳语` |

### English Accents

Only meaningful for English synthesis.

- `american accent`
- `british accent`
- `australian accent`
- `chinese accent`
- `canadian accent`
- `indian accent`
- `korean accent`
- `portuguese accent`
- `russian accent`
- `japanese accent`

### Chinese Dialects

Only meaningful for Chinese synthesis.

- `河南话`
- `陕西话`
- `四川话`
- `贵州话`
- `云南话`
- `桂林话`
- `济南话`
- `石家庄话`
- `甘肃话`
- `宁夏话`
- `青岛话`
- `东北话`

## Valid Examples

English:

```text
female, young adult, high pitch, british accent
male, elderly, low pitch
female, whisper
```

Chinese:

```text
女，青年，高音调，四川话
男，老年，低音调
女，耳语
```

Mixed language items are accepted if they do not violate the accent/dialect rule:

```text
female, young adult, 四川话
```

## Invalid Examples

These should be rejected by the port:

- `male, female`
- `british accent, 河南话`
- unsupported typo-only items with no valid match

## Practical Notes

- you can omit categories you do not care about
- one item per category is allowed
- if a CJK item is present, the normalized output uses Chinese separators
- if an accent is present, the port normalizes the instruction to English

The implementation lives in [frontend/voice_design.rs](../crates/omnivoice-infer/src/frontend/voice_design.rs).

