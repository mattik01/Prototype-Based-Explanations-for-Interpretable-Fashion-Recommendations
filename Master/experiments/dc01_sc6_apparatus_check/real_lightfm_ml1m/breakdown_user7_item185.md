# Why item 185 (1213 · Crime|Drama · GoodFellas (1990)) for user 7? — LightFM-style CBF (lightfm_tags)

Score **S = +0.3575**.

## exact per-feature contribution u·e_r

| line | contribution |
|---|---|
| tags = imdb top 250 | +0.0255 |
| tags = great acting | +0.0226 |
| tags = violence | +0.0205 |
| tags = violent | +0.0192 |
| tags = oscar (best directing) | +0.0174 |
| tags = classic | +0.0162 |
| tags = mentor | +0.0152 |
| tags = powerful ending | +0.0142 |
| tags = drama | +0.0138 |
| tags = excellent script | +0.0136 |
| tags = hit men | +0.0127 |
| tags = mob | +0.0121 |
| tags = dark humor | +0.0118 |
| tags = dialogue | +0.0116 |
| tags = gangster | +0.0107 |
| tags = anti-hero | -0.0105 |
| tags = good soundtrack | +0.0104 |
| tags = adapted from:book | +0.0102 |
| tags = narrated | +0.0097 |
| tags = gangsters | +0.0091 |
| tags = great movie | +0.0079 |
| genres = Crime | +0.0077 |
| tags = quotable | +0.0071 |
| tags = interesting | +0.0069 |
| tags = bloody | -0.0061 |
| tags = visceral | +0.0060 |
| tags = oscar (best picture) | -0.0057 |
| tags = fast paced | +0.0056 |
| tags = good acting | +0.0054 |
| tags = mafia | +0.0052 |
| tags = great dialogue | +0.0052 |
| tags = brutality | +0.0050 |
| tags = crime gone awry | +0.0048 |
| tags = original | +0.0047 |
| tags = storytelling | +0.0047 |
| tags = masterpiece | +0.0047 |
| tags = gritty | -0.0046 |
| tags = great ending | +0.0045 |
| tags = crime | +0.0034 |
| tags = based on a book | +0.0033 |
| tags = based on book | +0.0030 |
| tags = oscar (best actor) | +0.0026 |
| tags = stylized | +0.0025 |
| tags = highly quotable | +0.0023 |
| genres = Drama | +0.0021 |
| tags = foul language | +0.0021 |
| tags = good | -0.0020 |
| tags = organized crime | +0.0020 |
| tags = gratuitous violence | -0.0018 |
| tags = heist | +0.0016 |
| tags = rags to riches | -0.0015 |
| tags = stylish | +0.0009 |
| tags = amazing cinematography | +0.0009 |
| tags = based on true story | +0.0005 |
| tags = oscar (best supporting actor) | +0.0005 |
| tags = brutal | +0.0004 |

## Mechanism

No prototype layer: feature-level attribution is this model's entire explanation mechanism. The absence of a prototype-shaped reading is part of the comparison.

## Method notes

- Contributions are exact additive summands of the score S = u·(Σ_r e_r).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: n/a (no prototype layer).
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
