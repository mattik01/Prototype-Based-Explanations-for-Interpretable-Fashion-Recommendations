---
date: 2026-09-08
time: "16:20"
phase: 4
---
# Presentation of explanations to end users: out of thesis scope, cite the literature

**Decision (Matteo, 2026-09-08).** How the model's outputs are presented to a shopper in a
convincing way is NOT part of the thesis. The thesis is about an interpretable model; what is
done with its outputs downstream is not our concern. The transparency/scrutability literature
surfaced by the fU user-vocabulary research (angle D, 2026-09-07) is to be CITED as the place
where that question is handled, not built upon. No renderer changes (item examples under words,
deviation-from-crowd display) are planned as thesis work.

**What the literature says (for the citation paragraph).** Every scrutable user profile in this
line is a frequency-weighted mean of attributes plus a deviation or shrinkage step plus
sparsification to a few units, rendered with a concrete example item. Human evidence: in
Balog, Radlinski & Arakelyan (SIGIR 2019; 122 crowd users, top-5 statements) the example item is
the single most-endorsed unit, single-tag statements are accepted more than two-tag statements
(57 % vs 34 % full agreement), and 19–27 % rejected even that the example belonged to the
statement. Vig, Sen & Riedl (Tagsplanations, IUI 2009) found the user-to-tag preference is what
justifies, tag-to-item relevance is the organising order, and plain genre-level words beat
specific ones. Balog & Radlinski (SIGIR 2020): scrutability is the explanation goal least
correlated with the others — a more scrutable form is not necessarily felt as better. Zhang et
al. EFM (SIGIR 2014): feature-level explanations lifted CTR in an online A/B. Radlinski et al.
(SIGIR 2022) define scrutable as short enough to review and edit directly. Tintarev & Masthoff
(2007): seven explanation aims that can conflict. Zhang & Chen (FnTIR 2020): the three user-side
explanation styles are relevant users/items, features, and topic word clusters.

**Citation list (verified full text or abstract, per angle_D.md / angle_E.md):**
Balog, Radlinski, Arakelyan 2019 (doi 10.1145/3331184.3331211); Balog & Radlinski 2020
(10.1145/3397271.3401032); Radlinski, Balog, Diaz, Dixon, Wedin 2022 (10.1145/3477495.3531873);
Vig, Sen, Riedl 2009 (IUI); Zhang, Lai, Zhang, Zhang, Liu, Ma 2014 EFM (10.1145/2600428.2609579);
Tintarev & Masthoff 2007 (ICDE Workshops); Zhang & Chen 2020 (FnTIR 14(1), arXiv 1804.11192);
Herlocker, Konstan, Riedl 2000 (CSCW, Table 1); Chang, Harper, Terveen 2015 (CSCW, abstract only);
Mysore, McCallum, Zamani LACE 2023 (SIGIR, arXiv 2304.04250). Source notes:
`Master/temp/user_vocabulary_research_2026-09-07/angle_D.md`, `angle_E.md`.

**Thesis placement.** fU's contract (linear-editable profile of attribute words plus a disclosed
non-scrutable ID residual) is the object these papers evaluate; the thesis states the model is
compatible with that presentation line and points to it. Human-study numbers are from movie
tagging, not fashion; quote as motivation only, never as a transfer claim.

[[explanations]] [[thesis-writing]] [[phase-4]] [[decisions]]
