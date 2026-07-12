---
date: 2026-07-12
time: "18:30"
phase: 4
---
# M5′ read precisely: noise-for-crowding, the ids-arm division of labor, the M6′ link — and the centering option (K6′)

*(Verbatim-capture entry — dc05 review sitting, the M5′ deep-dive. User: "record
the option, it is very interesting, and otherwise record this stuff for the
discussion of ID vs. no-ID for the fU candidate." This entry is source
material for exactly that ids-vs-noid discussion; the design doc carries the
same content as dated amendments (§3.8 M5′ + K6′, 4c table).)*

## The mechanism, stated more carefully than the slogan

Averaging a basket does NOT pull a user toward the global average — it pulls
them toward **their own taste centroid**. A 3-item basket is 3 noisy draws
from the user's taste; a 200-item basket has averaged the noise away and sits
almost exactly at that user's personal mean direction. That part is law of
large numbers, and it is *good* — noise reduction.

The risk enters through a second fact: **personal centroids are crowded.**
Everyone's taste distribution shares the population-common mass (Solid
basics, the big departments), so heavy users converge *precisely* onto
centroids that sit angularly close to everyone else's. The C9 toy simulated
the worst case (all users drawing from ONE shared distribution — full
collapse); reality is in between. M5′ properly read: **averaging trades noise
for crowding — thin users are noisy but far apart; heavy users are precise
but close together.**

## The ids-arm division of labor (central input to the ids-vs-noid discussion)

The ID row's *relative weight* is constant across users (mean normalization),
but its *training quality* is not: a heavy user's ID row received hundreds of
gradient steps. So the ids arm has a built-in handover that strengthens
exactly as M5′ bites:

> **Words carry your taste class; the ID row carries your idiosyncrasy — and
> the ID's quality grows with history length precisely as the basket's
> individuating power fades.** Thin user: informative basket, useless ID.
> Heavy user: crowded basket, precise ID.

Consequences: M5′ primarily threatens the **noid arm** and any claim that
words alone should carry personalization; for the headline ids config it is a
re-routing of where individuality lives (with the honest cost that the ID
part of the explanation is the opaque line). Predicted ids-vs-noid signature:
the gap **widens specifically in the heavy history band** — this is the M5′
fingerprint to look for when interpreting the ablation pair, alongside the
existing gap-interpretation rule.

## The systemic link: M5′ feeds M6′

If composed user vectors crowd together, activations flatten across users,
and the loss's cheapest way to differentiate items becomes the
user-independent channel — mean-regression pushes score mass into B(t). The
instruments interlock deliberately: composed-cloud spread by history stratum,
activation spread across users, B(t) variance share. If M5′ is real, all
three move together.

## The recorded option: K6′ population-mean centering (unstacked)

Compose, then subtract the population mean-basket direction before the
cosine: q̃_u = q_u − m̄ (m̄ train-computed, buffered). The representation
becomes "how your basket DEVIATES from the average shopper's" — attacking
M5′ crowding and the M2′ omnipresence tilt with one move, at trivial cost.
It is a real mechanism change, not a loss knob: the exact-share claims
(C1′/C2′) must be re-derived (shares acquire a −m̄ term), and the explanation
sentences change meaning ("you buy more denim *than average*" — arguably an
even better explanation vocabulary). NOT stacked (R8); evidence trigger =
the M5′ instrument set reading badly on the heavy band. Registered in the
design doc as K6′ (§3.8) with the normalization exponent as the weaker lever.

[[phase-4]] [[decisions]] [[explanations]] [[experiments]]
