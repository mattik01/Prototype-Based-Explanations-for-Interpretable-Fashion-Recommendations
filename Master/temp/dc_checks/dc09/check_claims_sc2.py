#!/usr/bin/env python
"""check_claims_sc2.py -- black-box toy verification of the dc09 SC.2 claims D1-D11.

The merged model is implemented from claims_spec_sc2.md ALONE (no project code is
imported or read). CPU, float64 by default (float32 only where a claim says
bit-identical), toy sizes, seconds of runtime. Writes report_sc2.md next to itself.

Implementation conventions (stated in the report):
  * guarded cosine: cos(a,b) = a.b / (max(|a|,eps) * max(|b|,eps)), eps = 1e-8
  * a training row (u, i+, negs) uses ONE q_u^LOO (basket H_u \\ {i+}) for every
    candidate in the row (positive and negatives), for both user roles
  * inclusion regularisers are included in the gradient tests with fixed weights
    (0.3 R_proto_item - 0.2 R_batch_item + 0.25 R_proto_user + 0.15 R_batch_user),
    computed over the batch of candidate compositions / the batch of q_u^LOO
  * "machine precision" = relative error <= 1e-11 (float64), relative to the
    magnitude of the operands that enter the computation (cancellation-aware)
"""
import math
import os
from decimal import Decimal, getcontext
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-8
TOL = 1e-11
DIFF = 1e-9   # "differs": well above the float64 noise floor (incl. kappa-amplified rounding)
SEEDS = (0, 1, 2, 3, 4, 5)
LOSSES = ("softmax", "bce", "bpr")
COMP_KEYS = ("S", "B_item", "U_half", "B_user", "I_half")
PW_KEYS = ("P_u", "P_t", "W_u", "W_t")
SHAPES = (
    dict(name="A", sizes=(4, 5, 3), M=14, N=7, d=6, Ku=3, Kt=4),
    dict(name="B", sizes=(6, 7), M=20, N=9, d=4, Ku=5, Kt=2),
    dict(name="C", sizes=(3, 3, 4, 2), M=11, N=6, d=9, Ku=2, Kt=6),
)
SCALE_SETS = {
    "unit": dict(E_w=1.0, E_id=1.0, P=1.0, W_u=1.0, W_t=1.0),
    "small": dict(E_w=1e-3, E_id=1e-3, P=1e-3, W_u=1e-3, W_t=1e-3),
    "large": dict(E_w=1e3, E_id=1e3, P=1e3, W_u=1e3, W_t=1e3),
    "mixA": dict(E_w=1e3, E_id=1e-3, P=1.0, W_u=1e-3, W_t=1e1),
    "mixB": dict(E_w=1e-3, E_id=1e3, P=1e3, W_u=1e3, W_t=1e-3),
    "neg": dict(E_w=1.0, E_id=1.0, P=1.0, W_u=1.0, W_t=1.0),  # every entry negative
    "rand": None,  # per-block scale drawn from SCALE_GRID
}
SCALE_NAMES = tuple(SCALE_SETS)
SCALE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
RO_KEYS = ("word_that", "word_Bitem", "id_that", "id_Bitem", "word_cosshare", "id_cosshare", "naming")


# ============================================================== bookkeeping
class Rec:
    def __init__(self):
        self.r = {}

    def _g(self, k, kind):
        if k not in self.r:
            self.r[k] = dict(kind=kind, n=0, fail=0, max_err=0.0, max_raw=0.0,
                             min_diff=math.inf, max_diff=0.0, skipped=0)
        return self.r[k]

    def ident(self, k, err, tol=TOL, raw=None):
        g = self._g(k, "ident")
        g["n"] += 1
        err = float(err)
        if math.isnan(err):
            err = math.inf
        g["max_err"] = max(g["max_err"], err)
        if raw is not None:
            g["max_raw"] = max(g["max_raw"], float(raw))
        if not err <= tol:
            g["fail"] += 1

    def differ(self, k, dv, thr=DIFF):
        g = self._g(k, "differ")
        g["n"] += 1
        dv = float(dv)
        if math.isnan(dv):
            dv = 0.0
        g["min_diff"] = min(g["min_diff"], dv)
        g["max_diff"] = max(g["max_diff"], dv)
        if not dv > thr:
            g["fail"] += 1

    def check(self, k, ok):
        g = self._g(k, "check")
        g["n"] += 1
        if not ok:
            g["fail"] += 1

    def skip(self, k, kind="differ"):
        self._g(k, kind)["skipped"] += 1

    def ok(self, *ks):
        return all(k in self.r and self.r[k]["n"] > 0 and self.r[k]["fail"] == 0 for k in ks)

    def anyfail(self, *ks):
        return any(k in self.r and self.r[k]["fail"] > 0 for k in ks)

    def s(self, *ks):
        out = []
        for k in ks:
            g = self.r.get(k)
            if g is None:
                out.append(f"`{k}`: no cases")
                continue
            if g["kind"] == "ident":
                t = f"{g['n']} cases, max err {g['max_err']:.2e}"
                if g["max_raw"]:
                    t += f" (raw, un-normalised {g['max_raw']:.2e})"
                t += f", fails {g['fail']}"
            elif g["kind"] == "differ":
                t = (f"{g['n']} cases, rel. difference min {g['min_diff']:.2e} / max {g['max_diff']:.2e}, "
                     f"non-differing {g['fail']}")
            else:
                t = f"{g['n']} cases, failures {g['fail']}"
            if g["skipped"]:
                t += f", skipped {g['skipped']} (saturated/degenerate)"
            out.append(f"`{k}`: {t}")
        return "<br>".join(out)


REC = Rec()
SECTIONS = []


class Section:
    def __init__(self, cid, title):
        self.cid, self.title, self.items, self.notes = cid, title, [], []
        SECTIONS.append(self)

    def add(self, sub, verdict, detail, reading=None):
        self.items.append((sub, verdict, detail, reading))

    def note(self, s):
        self.notes.append(s)

    def overall(self):
        bad = [it[0] for it in self.items if it[1] == "FALSIFIED"]
        unt = [it[0] for it in self.items if it[1].startswith("UNTESTABLE")]
        if bad:
            return "FALSIFIED in " + ", ".join(bad)
        if unt:
            return "UNTESTABLE-AT-TOY-SCALE in " + ", ".join(unt)
        return "CONFIRMED"


def V(ok):
    return "CONFIRMED" if ok else "FALSIFIED"


# ============================================================== numerics helpers
def _mx(t):
    t = torch.as_tensor(t)
    return float(t.abs().max()) if t.numel() else 0.0


def rel(a, b, *mags):
    """max|a-b| / max(|a|,|b|,mags...) (operand-magnitude-aware relative error)."""
    a = torch.as_tensor(a).to(torch.float64)
    b = torch.as_tensor(b).to(torch.float64)
    num = _mx(a - b)
    if num == 0.0:
        return 0.0
    den = max([_mx(a), _mx(b)] + [float(m) for m in mags])
    return num / den if den > 0 else math.inf


def rowrel(gs, gi, gu):
    """row-by-row max |gs - (gi+gu)| / max(|gs|,|gi|,|gu|) for word-row gradients."""
    num = (gs - gi - gu).abs().amax(1)
    den = torch.stack([gs.abs().amax(1), gi.abs().amax(1), gu.abs().amax(1)]).amax(0)
    r = torch.where(num == 0, torch.zeros_like(num), num / den.clamp_min(1e-300))
    return float(r.max()) if r.numel() else 0.0


def nz(t):
    return _mx(t) > 0.0


# ============================================================== data
class Data:
    def __init__(self, rng, sh, layout, basket=(1, 6), kmax=6, multiset=False):
        sizes, M, N = sh["sizes"], sh["M"], sh["N"]
        if layout == "bagms":  # bag layout with word multisets (repeated words allowed)
            multiset = True
        Vn = int(sum(sizes))
        perm = rng.permutation(Vn)
        fields, off = [], 0
        for s in sizes:
            fields.append(np.sort(perm[off:off + s]))
            off += s
        C = np.zeros((M, Vn))
        for i in range(M):
            for ws in fields:
                if layout == "fixed":
                    C[i, rng.choice(ws)] += 1
                else:
                    if multiset:
                        k = int(rng.integers(0, kmax + 1))
                        ch = rng.choice(ws, size=k, replace=True)
                    else:
                        k = int(rng.integers(0, min(kmax, len(ws)) + 1))
                        ch = rng.choice(ws, size=k, replace=False)
                    np.add.at(C[i], ch, 1)
        lo, hi = basket[0], min(basket[1], M)
        baskets = [sorted(rng.choice(M, size=int(rng.integers(lo, hi + 1)), replace=False).tolist())
                   for _ in range(N)]
        self._set(fields, C, baskets, layout)

    @classmethod
    def manual(cls, sizes, C, baskets, layout):
        D = cls.__new__(cls)
        fields, off = [], 0
        for s in sizes:
            fields.append(np.arange(off, off + s))
            off += s
        D._set(fields, np.asarray(C, dtype=float), baskets, layout)
        return D

    def _set(self, fields, C, baskets, layout):
        self.fields = [np.asarray(f, dtype=np.int64) for f in fields]
        self.fields_t = [torch.tensor(f) for f in self.fields]
        self.V = int(sum(len(f) for f in fields))
        self.F = len(fields)
        fo = np.empty(self.V, dtype=np.int64)
        for phi, ws in enumerate(self.fields):
            fo[ws] = phi
        self.field_of = torch.tensor(fo)
        self.C = torch.tensor(np.asarray(C), dtype=torch.float64)
        self.M = self.C.shape[0]
        self.baskets = [list(b) for b in baskets]
        self.N = len(self.baskets)
        self.layout = layout
        self.K = torch.stack([self.C[:, f].sum(1) for f in self.fields_t], 1)  # M x F per-field counts
        self._omf = None

    def omega(self, basket):
        if len(basket) == 0:
            return torch.zeros(self.V)
        return self.C[list(basket)].sum(0) / len(basket)

    def om_full(self):
        if self._omf is None:
            self._omf = torch.stack([self.omega(H) for H in self.baskets])
        return self._omf

    def set_basket(self, u, H):
        self.baskets[u] = list(H)
        self._omf = None

    def words(self, i):
        return torch.nonzero(self.C[i]).flatten().tolist()


class Rows:
    """training rows (u, i+, negs) with the LOO user weights."""

    def __init__(self, D, triples):
        self.list = [(int(u), int(ip), [int(x) for x in ng]) for (u, ip, ng) in triples]
        self.users = torch.tensor([t[0] for t in self.list], dtype=torch.long)
        self.Om = torch.stack([D.omega([j for j in D.baskets[u] if j != ip]) for (u, ip, _) in self.list])
        self.Omfull = D.om_full()[self.users]
        self.cand = torch.tensor([[ip] + ng for (u, ip, ng) in self.list], dtype=torch.long)

    @classmethod
    def sample(cls, D, rng, n_neg=3, min_H=1):
        T = []
        for u in range(D.N):
            H = D.baskets[u]
            if len(H) < min_H:
                continue
            pool = [j for j in range(D.M) if j not in H] or list(range(D.M))
            for ip in H:
                T.append((u, ip, rng.choice(pool, size=n_neg, replace=len(pool) < n_neg).tolist()))
        return cls(D, T)


# ============================================================== parameters
def blk(rng, shape, scale, neg=False):
    x = rng.standard_normal(shape) * scale
    if neg:
        x = -np.abs(x)
    return torch.tensor(x)


def scale_cfg(name, rng):
    if name == "rand":
        sc = {k: float(rng.choice(SCALE_GRID)) for k in ("E_w", "E_id", "P", "W_u", "W_t")}
    else:
        sc = dict(SCALE_SETS[name])
    sc["neg"] = name == "neg"
    return sc


def make_params(rng, D, sh, table, scn):
    sc = scale_cfg(scn, rng)
    ng = sc["neg"]
    d, Ku, Kt = sh["d"], sh["Ku"], sh["Kt"]
    p = dict(P_u=blk(rng, (Ku, d), sc["P"], ng), P_t=blk(rng, (Kt, d), sc["P"], ng),
             W_u=blk(rng, (Kt, d), sc["W_u"], ng), W_t=blk(rng, (Ku, d), sc["W_t"], ng))
    Ew = blk(rng, (D.V, d), sc["E_w"], ng)
    Ei = blk(rng, (D.M, d), sc["E_id"], ng)
    Eu = blk(rng, (D.N, d), sc["E_id"], ng)
    if table == "shared":
        p.update(E_w=Ew, E_Iid=Ei, E_Uid=Eu)
    else:
        p.update(E_I=torch.cat([Ew, Ei]), E_U=torch.cat([blk(rng, (D.V, d), sc["E_w"], ng), Eu]))
    return p, sc


def tie(ps):
    """sep model at the tied point E_I[0:V] = E_U[0:V] = E_w (numerically tied copies)."""
    return dict(P_u=ps["P_u"].clone(), P_t=ps["P_t"].clone(), W_u=ps["W_u"].clone(), W_t=ps["W_t"].clone(),
                E_I=torch.cat([ps["E_w"], ps["E_Iid"]]), E_U=torch.cat([ps["E_w"], ps["E_Uid"]]))


def tabs(p, Vn):
    """(item word rows, item ID rows, user word rows, user ID rows)."""
    if "E_w" in p:
        return p["E_w"], p["E_Iid"], p["E_w"], p["E_Uid"]
    return p["E_I"][:Vn], p["E_I"][Vn:], p["E_U"][:Vn], p["E_U"][Vn:]


def emb_keys(p):
    return [k for k in p if k.startswith("E")]


def ratio(p):
    return float(p["W_t"].norm() / p["W_u"].norm())


# ============================================================== model
def comp_items(p, D, items, arm, C=None):
    Ewi, Eidi, _, _ = tabs(p, D.V)
    items = torch.as_tensor(items, dtype=torch.long)
    q = (D.C if C is None else C)[items].to(Ewi.dtype) @ Ewi
    return q + Eidi[items] if arm == "ids" else q


def comp_users(p, D, users, Om, arm):
    _, _, Ewu, Eidu = tabs(p, D.V)
    users = torch.as_tensor(users, dtype=torch.long)
    q = Om.to(Ewu.dtype) @ Ewu
    return q + Eidu[users] if arm == "ids" else q


def cosm(a, P):
    na = torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(EPS)
    nP = torch.linalg.vector_norm(P, dim=-1).clamp_min(EPS)
    return (a @ P.T) / na / nP


def uroles(p, qu):
    return 1.0 + cosm(qu, p["P_u"]), qu @ p["W_u"].T  # u*, u^


def iroles(p, qi):
    return 1.0 + cosm(qi, p["P_t"]), qi @ p["W_t"].T  # t*, t^


def grid(p, D, arm, qu=None, qi=None, Om=None):
    """inference grid (full baskets unless qu given): S and the four components."""
    if qi is None:
        qi = comp_items(p, D, torch.arange(D.M), arm)
    if qu is None:
        qu = comp_users(p, D, torch.arange(D.N), D.om_full() if Om is None else Om, arm)
    us, uh = uroles(p, qu)
    ts, th = iroles(p, qi)
    S = us @ th.T + uh @ ts.T
    return dict(S=S, B_item=th.sum(1)[None, :].expand_as(S).clone(), U_half=(us - 1.0) @ th.T,
                B_user=uh.sum(1)[:, None].expand_as(S).clone(), I_half=uh @ (ts - 1.0).T,
                qu=qu, qi=qi, us=us, uh=uh, ts=ts, th=th)


def fwd_rows(p, D, arm, rows, detach=None, qu=None, full=False):
    """training rows: one q_u^LOO per row feeds both user roles for all candidates."""
    if qu is None:
        qu = comp_users(p, D, rows.users, rows.Omfull if full else rows.Om, arm)
    R, c = rows.cand.shape
    qc = comp_items(p, D, rows.cand.reshape(-1), arm).reshape(R, c, -1)
    us, uh = uroles(p, qu)
    qs = qc.detach() if detach == "tstar" else qc   # 'tstar' detached -> only t^ path
    qh = qc.detach() if detach == "that" else qc    # 'that' detached  -> only t* path
    ts = 1.0 + cosm(qs, p["P_t"])
    th = qh @ p["W_t"].T
    S = torch.einsum("rk,rck->rc", us, th) + torch.einsum("rk,rck->rc", uh, ts)
    return dict(S=S, qu=qu, qc=qc, us=us, uh=uh, ts=ts, th=th)


def regs(q, P):
    Cm = cosm(q, P)
    return Cm.max(0).values.sum(), Cm.max(1).values.sum()  # R_proto, R_batch


def loss_of(S, kind):
    if kind == "softmax":
        return -torch.log_softmax(S, 1)[:, 0].sum()
    # exact log-sigmoid (torch's softplus switches to an exact-linear approximation above 20)
    if kind == "bce":
        return -Fn.logsigmoid(S[:, 0]).sum() - Fn.logsigmoid(-S[:, 1:]).sum()
    if kind == "bpr":
        return -Fn.logsigmoid(S[:, :1] - S[:, 1:]).sum()
    raise ValueError(kind)


def total(p, D, arm, rows, kind, reg=True, detach=None):
    o = fwd_rows(p, D, arm, rows, detach)
    L = loss_of(o["S"], kind)
    if reg:
        a1, a2 = regs(o["qc"].reshape(-1, o["qc"].shape[-1]), p["P_t"])
        b1, b2 = regs(o["qu"], p["P_u"])
        L = L + 0.3 * a1 - 0.2 * a2 + 0.25 * b1 + 0.15 * b2
    return L


def grads(p, fn):
    q = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
    L = fn(q)
    gs = torch.autograd.grad(L, list(q.values()), allow_unused=True)
    return L.detach(), {k: (torch.zeros_like(q[k]) if g is None else g) for k, g in zip(q, gs)}


def iter_cfg(layouts=("fixed", "bag", "bagms"), arms=("ids", "noid"), seeds=SEEDS, shapes=SHAPES,
             scales=SCALE_NAMES, tag=0):
    for s in seeds:
        for si, sh in enumerate(shapes):
            for li, lay in enumerate(layouts):
                for ai, arm in enumerate(arms):
                    for ci, scn in enumerate(scales):
                        rng = np.random.default_rng([tag, s, si, li, ai, ci])
                        yield s, sh, lay, arm, scn, rng


def d5_map(p, D, v, item_shift=None, user_shift=None, sides=("I", "U")):
    """word rows += v[field(f)]; item IDs -= item_shift (default sum_phi v_phi); user IDs -= user_shift."""
    sv = v.sum(0)
    ish = sv.expand(D.M, -1) if item_shift is None else item_shift
    ush = sv.expand(D.N, -1) if user_shift is None else user_shift
    Wv = v[D.field_of]
    q = {k: x.clone() for k, x in p.items()}
    if "E_w" in p:
        q["E_w"] = p["E_w"] + Wv
        q["E_Iid"] = p["E_Iid"] - ish
        q["E_Uid"] = p["E_Uid"] - ush
    else:
        Vn = D.V
        if "I" in sides:
            q["E_I"] = torch.cat([p["E_I"][:Vn] + Wv, p["E_I"][Vn:] - ish])
        if "U" in sides:
            q["E_U"] = torch.cat([p["E_U"][:Vn] + Wv, p["E_U"][Vn:] - ush])
    return q


def kappa(p, p2, D, arm, rows=None):
    """conditioning factor: operand magnitude / smallest composition magnitude."""
    big = max(_mx(x) for x in [p[k] for k in emb_keys(p)] + [p2[k] for k in emb_keys(p2)])
    qi = comp_items(p, D, torch.arange(D.M), arm)
    qu = comp_users(p, D, torch.arange(D.N), D.om_full(), arm)
    small = min(float(qi.abs().amax(1).min()), float(qu.abs().amax(1).min()))
    if rows is not None:
        small = min(small, float(comp_users(p, D, rows.users, rows.Om, arm).abs().amax(1).min()))
    return big / max(small, 1e-300), big


def flat(p, keys):
    return torch.cat([p[k].reshape(-1) for k in keys])


# ============================================================== D1
def test_D1():
    sec = Section("D1", "Sharing keystone -- chain rule")
    ncfg = 0
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=1):
        ncfg += 1
        D = Data(rng, sh, lay)
        rows = Rows.sample(D, rng, 3)
        ps, sc = make_params(rng, D, sh, "shared", scn)
        Vn = D.V
        for dt, tg in ((torch.float64, "f64"), (torch.float32, "f32")):
            pa = {k: v.to(dt) for k, v in ps.items()}
            pb = tie(pa)
            ga, gb = grid(pa, D, arm), grid(pb, D, arm)
            REC.check(f"D1a.S+4components.bitwise.{tg}", all(torch.equal(ga[k], gb[k]) for k in COMP_KEYS))
            oa, ob = fwd_rows(pa, D, arm, rows), fwd_rows(pb, D, arm, rows)
            REC.check(f"D1a.LOO-training-scores.bitwise.{tg}", torch.equal(oa["S"], ob["S"]))
            ra = (regs(ga["qi"], pa["P_t"]) + regs(ga["qu"], pa["P_u"])
                  + regs(oa["qc"].reshape(-1, sh["d"]), pa["P_t"]) + regs(oa["qu"], pa["P_u"]))
            rb = (regs(gb["qi"], pb["P_t"]) + regs(gb["qu"], pb["P_u"])
                  + regs(ob["qc"].reshape(-1, sh["d"]), pb["P_t"]) + regs(ob["qu"], pb["P_u"]))
            REC.check(f"D1a.regularisers.bitwise.{tg}", all(torch.equal(x, y) for x, y in zip(ra, rb)))
        g = grid(ps, D, arm)
        mag = max(_mx(g[k]) for k in COMP_KEYS)
        REC.ident("D1.S=sum-of-4-components", rel(g["S"], g["B_item"] + g["U_half"] + g["B_user"] + g["I_half"], mag))
        pt = tie(ps)
        for kind in LOSSES:
            for reg in (False, True):
                Ls, gs = grads(ps, lambda q: total(q, D, arm, rows, kind, reg))
                Lt, gt = grads(pt, lambda q: total(q, D, arm, rows, kind, reg))
                REC.check("D1b.loss-value.bitwise", torch.equal(Ls, Lt))
                REC.ident(f"D1b.grad E_w = gE_I[:V]+gE_U[:V] (row by row).{kind}",
                          rowrel(gs["E_w"], gt["E_I"][:Vn], gt["E_U"][:Vn]))
                REC.ident(f"D1b.grad ID blocks.{kind}",
                          max(rel(gs["E_Iid"], gt["E_I"][Vn:]), rel(gs["E_Uid"], gt["E_U"][Vn:])))
                REC.ident(f"D1b.grad P,W.{kind}", max(rel(gs[k], gt[k]) for k in PW_KEYS))
                if not reg:
                    if nz(gt["E_I"][:Vn]) or nz(gt["E_U"][:Vn]):
                        REC.differ("D1c.tied point: gE_I[:V] != gE_U[:V]", rel(gt["E_I"][:Vn], gt["E_U"][:Vn]))
                    else:
                        REC.skip("D1c.tied point: gE_I[:V] != gE_U[:V]")
        # (c) generic untied sep point
        pu = tie(ps)
        pu["E_U"] = torch.cat([blk(rng, (Vn, sh["d"]), sc["E_w"], sc["neg"]), ps["E_Uid"]])
        _, gu = grads(pu, lambda q: total(q, D, arm, rows, "softmax", True))
        if nz(gu["E_I"][:Vn]) or nz(gu["E_U"][:Vn]):
            REC.differ("D1c.untied point: gE_I[:V] != gE_U[:V]", rel(gu["E_I"][:Vn], gu["E_U"][:Vn]))
        else:
            REC.skip("D1c.untied point: gE_I[:V] != gE_U[:V]")
        Su = grid(pu, D, arm)["S"]
        base = {k: pu[k] for k in PW_KEYS}
        cands = (pu["E_I"][:Vn], pu["E_U"][:Vn], 0.5 * (pu["E_I"][:Vn] + pu["E_U"][:Vn]))
        dmin = min(rel(Su, grid(dict(base, E_w=c, E_Iid=pu["E_I"][Vn:], E_Uid=pu["E_U"][Vn:]), D, arm)["S"])
                   for c in cands)
        REC.differ("D1c.untied point: no shared model (E_w in {E_I,E_U,avg}) reproduces S", dmin)

    ka = ["D1a.S+4components.bitwise.f64", "D1a.LOO-training-scores.bitwise.f64", "D1a.regularisers.bitwise.f64",
          "D1a.S+4components.bitwise.f32", "D1a.LOO-training-scores.bitwise.f32", "D1a.regularisers.bitwise.f32"]
    sec.add("(a)", V(REC.ok(*ka)), REC.s(*ka, "D1.S=sum-of-4-components"),
            "sep at the tied point (E_I=[E_w;E_Iid], E_U=[E_w;E_Uid]) vs shared, same arithmetic graph; "
            "inference grid (full baskets), LOO training rows and both inclusion regularisers on both sides, "
            "torch.equal (bitwise) in float64 and float32.")
    kb = ["D1b.loss-value.bitwise"] + [f"D1b.grad E_w = gE_I[:V]+gE_U[:V] (row by row).{k}" for k in LOSSES] + \
         [f"D1b.grad ID blocks.{k}" for k in LOSSES] + [f"D1b.grad P,W.{k}" for k in LOSSES]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb),
            "each loss with and without the inclusion regularisers, LOO active; per-row error normalised by "
            "max(|g_shared|,|g_I|,|g_U|) of that row.")
    kc = ["D1c.tied point: gE_I[:V] != gE_U[:V]", "D1c.untied point: gE_I[:V] != gE_U[:V]",
          "D1c.untied point: no shared model (E_w in {E_I,E_U,avg}) reproduces S"]
    sec.add("(c)", V(REC.ok(*kc)), REC.s(*kc),
            "sanity read: the two word-row gradients differ (at untied AND at tied points, so the shared gradient "
            "is not 2x either one), and no shared model with E_w in {E_I, E_U, mean} reproduces an untied sep model.")
    sec.note(f"{ncfg} configurations = 6 seeds x 3 shapes x {{fixed,bag,bagms}} x {{ids,noid}} x 7 scale sets "
             f"(unit, small 1e-3, large 1e3, two mixed, all-negative, random per block in 1e-3..1e3).")
    sec.note("Bit-identity holds because both models run the same op sequence on bit-equal values. A different "
             "but equivalent reduction order (e.g. computing the shared q_u via sum_j (q_j - e_ID(j))) is only "
             "equal to machine precision, not bitwise -- the claim is about the model, not every implementation.")


# ============================================================== D2
def find_row(D, rng, shared_word, n_neg=2):
    for u in rng.permutation(D.N).tolist():
        H = D.baskets[u]
        for ip in rng.permutation(H).tolist():
            others = [j for j in H if j != ip]
            wo = set(torch.nonzero(D.C[others].sum(0)).flatten().tolist()) if others else set()
            for f in D.words(ip):
                if (f in wo) != shared_word:
                    continue
                pool = [j for j in range(D.M) if j not in H and D.C[j, f] == 0]
                if len(pool) < n_neg:
                    continue
                return u, ip, f, rng.choice(pool, size=n_neg, replace=False).tolist()
    return None


def test_D2():
    sec = Section("D2", "LOO under sharing -- gradient routing")
    for s, sh, lay, arm, scn, rng in iter_cfg(arms=("ids",), tag=2):
        D = Data(rng, sh, lay)
        rows = Rows.sample(D, rng, 3)
        ps, sc = make_params(rng, D, sh, "shared", scn)
        o = fwd_rows(ps, D, "ids", rows)
        qf = comp_users(ps, D, rows.users, rows.Omfull, "ids")
        uh_bug = qf @ ps["W_u"].T
        dif = uh_bug - o["uh"]
        REC.ident("D2a.u^'-u^ = W_u(q_full - q_LOO)", rel(dif, (qf - o["qu"]) @ ps["W_u"].T, _mx(uh_bug), _mx(o["uh"])))
        structural = (rows.Om - rows.Omfull).abs().amax(1) == 0
        dn = dif.norm(dim=1) / torch.maximum(uh_bug.norm(dim=1), o["uh"].norm(dim=1)).clamp_min(1e-300)
        for r in range(len(rows.list)):
            if structural[r]:
                REC.skip("D2a.u^'-u^ != 0")
            else:
                REC.differ("D2a.u^'-u^ != 0", dn[r])
        # the buggy (cosine-only LOO) implementation changes the training scores
        S_bug = torch.einsum("rk,rck->rc", o["us"], o["th"]) + torch.einsum("rk,rck->rc", uh_bug, o["ts"])
        if (~structural).any():
            REC.differ("D2a.cosine-only-LOO implementation changes S", rel(S_bug[~structural], o["S"][~structural]))

    nb = nc = 0
    for s, sh, lay, arm, scn, rng in iter_cfg(arms=("ids", "noid"), tag=21):
        D = Data(rng, sh, lay)
        ps, sc = make_params(rng, D, sh, "shared", scn)
        pt = tie(ps)
        for want_shared in (False, True):
            hit = find_row(D, rng, want_shared)
            if hit is None:
                continue
            u, ip, f, negs = hit
            rows1 = Rows(D, [(u, ip, negs)])
            tag = "c" if want_shared else "b"
            if want_shared:
                nc += 1
            else:
                nb += 1
            for kind in LOSSES:
                for reg in (False, True):
                    _, gs = grads(ps, lambda q: total(q, D, arm, rows1, kind, reg))
                    _, gt = grads(pt, lambda q: total(q, D, arm, rows1, kind, reg))
                    if not want_shared:
                        REC.check("D2b.sep gE_U[f] exactly 0 (bitwise)", bool((gt["E_U"][f] == 0).all()))
                        REC.ident("D2b.shared gE_w[f] = sep gE_I[f]", rel(gs["E_w"][f], gt["E_I"][f]))
                        REC.check("D2b.(info) shared gE_w[f] == sep gE_I[f] bitwise", torch.equal(gs["E_w"][f], gt["E_I"][f]))
                    else:
                        if nz(gs["E_w"][f]) or nz(gt["E_I"][f]):
                            REC.differ("D2c.user-path gE_U[f] != 0", _mx(gt["E_U"][f]) / max(_mx(gt["E_I"][f]), _mx(gt["E_U"][f]), 1e-300))
                        else:
                            REC.skip("D2c.user-path gE_U[f] != 0")
                        REC.ident("D2c.shared gE_w[f] = gE_I[f] + gE_U[f]",
                                  rowrel(gs["E_w"][f:f + 1], gt["E_I"][f:f + 1], gt["E_U"][f:f + 1]))
                if not want_shared:
                    _, gfull = grads(ps, lambda q: total(q, D, arm, rows1, kind, False))
                    _, gts = grads(ps, lambda q: total(q, D, arm, rows1, kind, False, detach="that"))
                    _, gth = grads(ps, lambda q: total(q, D, arm, rows1, kind, False, detach="tstar"))
                    m = max(_mx(gfull["E_w"][f]), _mx(gts["E_w"][f]), _mx(gth["E_w"][f]))
                    if m == 0.0:
                        REC.skip(f"D2b.t* path to e_f nonzero [{arm}]")
                        REC.skip(f"D2b.t^ path to e_f nonzero [{arm}]")
                    else:
                        REC.differ(f"D2b.t* path to e_f nonzero [{arm}]", _mx(gts["E_w"][f]) / m, thr=1e-13)
                        REC.differ(f"D2b.t^ path to e_f nonzero [{arm}]", _mx(gth["E_w"][f]) / m, thr=1e-13)
                        if arm == "noid" and _mx(gts["E_w"][f]) == 0.0:
                            om = rows1.Om[0]
                            REC.check("D2b.(noid) zero t* path <=> q_u^LOO = 0 (empty/word-less LOO basket)", not nz(om))
                        REC.ident("D2b.t* path + t^ path = full", rel(gts["E_w"][f] + gth["E_w"][f], gfull["E_w"][f], m))
    # batch-level caveat: all rows of the same user in one batch
    rng = np.random.default_rng([2, 99])
    D = Data(rng, SHAPES[0], "fixed")
    ps, _ = make_params(rng, D, SHAPES[0], "shared", "unit")
    hit = None
    for _ in range(50):
        hit = find_row(D, rng, False)
        if hit is not None and len(D.baskets[hit[0]]) >= 2:
            break
    caveat = ""
    if hit is not None and len(D.baskets[hit[0]]) >= 2:
        u, ip, f, negs = hit
        allrows = Rows(D, [(u, j, negs) for j in D.baskets[u]])
        _, gt = grads(tie(ps), lambda q: total(q, D, "ids", allrows, "softmax", False))
        caveat = (f"Batch-level caveat (per-row claim unaffected): when the batch also contains the user's OTHER "
                  f"positives (u={u}, |H_u|={len(D.baskets[u])}), those rows' LOO baskets contain i+, so the user "
                  f"path into e_f is nonzero over the batch: max|gE_U[f]| = {_mx(gt['E_U'][f]):.3e}.")
    sec.add("(a)", V(REC.ok("D2a.u^'-u^ = W_u(q_full - q_LOO)", "D2a.u^'-u^ != 0",
                            "D2a.cosine-only-LOO implementation changes S")),
            REC.s("D2a.u^'-u^ = W_u(q_full - q_LOO)", "D2a.u^'-u^ != 0", "D2a.cosine-only-LOO implementation changes S"),
            "rows where the LOO and full word weights coincide structurally (omega_full = omega_LOO, e.g. a word-less "
            "positive in a |H|=1 basket, or all basket items with the same word multiset) give u^'-u^ = 0 exactly "
            "and are counted as skipped -- they are data-degenerate, not parameter-generic.")
    kb = ["D2b.sep gE_U[f] exactly 0 (bitwise)", "D2b.shared gE_w[f] = sep gE_I[f]", "D2b.t* path to e_f nonzero [ids]",
          "D2b.t^ path to e_f nonzero [ids]", "D2b.t* path + t^ path = full"]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb, "D2b.(info) shared gE_w[f] == sep gE_I[f] bitwise") +
            f"<br>constructed rows: {nb}<br>noid arm (outside the claim's 'shared ids' scope, for the record): " +
            REC.s("D2b.t* path to e_f nonzero [noid]", "D2b.t^ path to e_f nonzero [noid]",
                  "D2b.(noid) zero t* path <=> q_u^LOO = 0 (empty/word-less LOO basket)"),
            "single training row (u,i+,negs) with f in f_{i+}, f not in any other item of H_u, not in any sampled "
            "negative; all three losses, with/without regularisers; role split by detaching the item composition in "
            "one role at a time. Verdict on the ids arm (the claim's scope). In the noid arm the t* path into e_f is "
            "EXACTLY zero whenever q_u^LOO = 0 (|H_u| = 1 or a word-less LOO basket): then u^ = 0, so the I-half "
            "carries no gradient -- a noid-only degeneracy worth knowing for D2-style routing arguments.")
    kc = ["D2c.user-path gE_U[f] != 0", "D2c.shared gE_w[f] = gE_I[f] + gE_U[f]"]
    sec.add("(c)", V(REC.ok(*kc)), REC.s(*kc) + f"<br>constructed rows: {nc}")
    if caveat:
        sec.note(caveat)


# ============================================================== D3
def test_D3():
    sec = Section("D3", "Centroid identity")
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=3):
        D = Data(rng, sh, lay)
        ps, sc = make_params(rng, D, sh, "shared", scn)
        qi = comp_items(ps, D, torch.arange(D.M), arm)
        Ewi, Eidi, Ewu, Eidu = tabs(ps, D.V)
        idm = _mx(Eidi) if arm == "ids" else 0.0
        for u, H in enumerate(D.baskets):
            if arm == "ids":
                REC.ident("D3a.q_u^words = mean_j (q_j - e_ID(j))", rel(D.omega(H) @ Ewu, (qi[H] - Eidi[H]).mean(0), idm))
            else:
                qu = comp_users(ps, D, [u], D.omega(H)[None], "noid")[0]
                REC.ident("D3b.noid q_u = mean_j q_j", rel(qu, qi[H].mean(0)))
            for i in H:
                B = [j for j in H if j != i]
                lhs = comp_users(ps, D, [u], D.omega(B)[None], arm)[0]
                if arm == "ids":
                    rhs = ((qi[B] - Eidi[B]).mean(0) if B else torch.zeros(sh["d"])) + Eidu[u]
                else:
                    rhs = qi[B].mean(0) if B else torch.zeros(sh["d"])
                REC.ident(f"D3c.LOO form ({arm})", rel(lhs, rhs, idm, _mx(Eidu) if arm == "ids" else 0.0))
        # (d) sep, generic untied point
        pu, _ = make_params(rng, D, sh, "sep", scn)
        qi2 = comp_items(pu, D, torch.arange(D.M), arm)
        Ewi2, Eidi2, Ewu2, Eidu2 = tabs(pu, D.V)
        for u, H in enumerate(D.baskets):
            om = D.omega(H)
            if not nz(om):
                REC.skip(f"D3d.sep: identity fails ({arm})")
                continue
            if arm == "ids":
                REC.differ(f"D3d.sep: identity fails ({arm})", rel(om @ Ewu2, (qi2[H] - Eidi2[H]).mean(0)))
            else:
                REC.differ(f"D3d.sep: identity fails ({arm})", rel(om @ Ewu2, qi2[H].mean(0)))
            for i in H:
                B = [j for j in H if j != i]
                if not B or not nz(D.omega(B)):
                    continue
                lhs = D.omega(B) @ Ewu2
                rhs = (qi2[B] - Eidi2[B]).mean(0) if arm == "ids" else qi2[B].mean(0)
                REC.differ(f"D3d.sep: LOO identity fails ({arm})", rel(lhs, rhs))
    sec.add("(a)", V(REC.ok("D3a.q_u^words = mean_j (q_j - e_ID(j))")), REC.s("D3a.q_u^words = mean_j (q_j - e_ID(j))"),
            "'exactly' read as an algebraic identity verified to machine precision; the error is normalised by the "
            "ID-row magnitude because q_j - e_ID(j) cancels (raw relative error grows like |e_ID|/|words| x 1e-16). "
            "It is not bitwise in floating point.")
    sec.add("(b)", V(REC.ok("D3b.noid q_u = mean_j q_j")), REC.s("D3b.noid q_u = mean_j q_j"))
    sec.add("(c)", V(REC.ok("D3c.LOO form (ids)", "D3c.LOO form (noid)")), REC.s("D3c.LOO form (ids)", "D3c.LOO form (noid)"),
            "empty LOO basket (|H_u|=1) -> word part 0 on both sides, as in the spec.")
    kd = ["D3d.sep: identity fails (ids)", "D3d.sep: identity fails (noid)", "D3d.sep: LOO identity fails (ids)",
          "D3d.sep: LOO identity fails (noid)"]
    sec.add("(d)", V(REC.ok(*kd)), REC.s(*kd),
            "users whose basket has no words at all (bag layout) satisfy the identity trivially (0 = 0) and are "
            "skipped; every other user fails it at a generic sep point.")


# ============================================================== D4
def Tc(ps, D, c):
    """ratio-changing reparametrisation of the shared ids arm (full baskets)."""
    m = D.om_full() @ ps["E_w"]
    return dict(ps, E_w=ps["E_w"] / c, E_Iid=ps["E_Iid"] / c, E_Uid=c * ps["E_Uid"] + (c - 1.0 / c) * m,
                W_t=c * ps["W_t"], W_u=ps["W_u"] / c)


def test_D4():
    sec = Section("D4", "Gauge collapse under sharing")
    CS = (1e-3, 0.37, 2.5, 1e3)
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=4, seeds=SEEDS[:4]):
        D = Data(rng, sh, lay)
        rows = Rows.sample(D, rng, 3)
        pu, sc = make_params(rng, D, sh, "sep", scn)
        g0 = grid(pu, D, arm)
        o0 = fwd_rows(pu, D, arm, rows)
        for c in CS + (float(np.exp(2.0 * rng.normal())),):
            p2 = dict(pu, E_I=c * pu["E_I"], W_t=pu["W_t"] / c)
            p3 = dict(pu, E_U=c * pu["E_U"], W_u=pu["W_u"] / c)
            for pp, tg in ((p2, "item side (E_I,W_t)"), (p3, "user side (E_U,W_u)")):
                g = grid(pp, D, arm)
                o = fwd_rows(pp, D, arm, rows)
                REC.ident(f"D4a.sep per-side rescale, {tg}", max([rel(g[k], g0[k]) for k in COMP_KEYS] + [rel(o["S"], o0["S"])]))
            REC.ident("D4d.sep per-side: ratio' = ratio/c", rel(ratio(p2) * c, ratio(pu)))
            REC.differ("D4d.sep per-side: ratio changes", abs(ratio(p2) - ratio(pu)) / ratio(pu))
        nmax = float(g0["qi"].norm(dim=1).max())
        if nmax > 0 and nz(g0["uh"]):
            cg = 1e-2 * EPS / nmax
            p4 = dict(pu, E_I=cg * pu["E_I"], W_t=pu["W_t"] / cg)
            REC.differ("D4a.guard regime (c*|q_i| < eps): S changes", rel(grid(p4, D, arm)["S"], g0["S"]))
        ps, sc = make_params(rng, D, sh, "shared", scn)
        g0 = grid(ps, D, arm)
        o0 = fwd_rows(ps, D, arm, rows)
        for c in CS:
            pb = dict(ps, E_w=c * ps["E_w"], E_Iid=c * ps["E_Iid"], W_t=ps["W_t"] / c)
            gb = grid(pb, D, arm)
            REC.differ("D4b.shared partial rescale changes S", rel(gb["S"], g0["S"]))
            REC.differ("D4b.shared partial rescale changes u^", rel(gb["uh"], g0["uh"]))
            pc = dict(ps, E_w=c * ps["E_w"], E_Iid=c * ps["E_Iid"], E_Uid=c * ps["E_Uid"],
                      W_t=ps["W_t"] / c, W_u=ps["W_u"] / c)
            gc = grid(pc, D, arm)
            oc = fwd_rows(pc, D, arm, rows)
            REC.ident("D4c.shared joint rescale: S, 4 components, LOO scores unchanged",
                      max([rel(gc[k], g0[k]) for k in COMP_KEYS] + [rel(oc["S"], o0["S"])]))
            REC.ident("D4d.shared joint rescale: ratio unchanged", rel(ratio(pc), ratio(ps)))
            if arm == "ids":
                pT = Tc(ps, D, c)
                gT = grid(pT, D, arm)
                oT = fwd_rows(pT, D, arm, rows)
                m_ = D.om_full() @ ps["E_w"]
                opmag = torch.stack([(m_ / c).abs().amax(1), ((c - 1.0 / c) * m_).abs().amax(1),
                                     (c * ps["E_Uid"]).abs().amax(1)]).amax(0)
                kT = float((opmag / gT["qu"].abs().amax(1).clamp_min(1e-300)).max())
                eT = max(rel(gT[k], g0[k]) for k in COMP_KEYS)
                REC.ident("D4d(iii).T_c: full-basket S + 4 components unchanged", eT / max(1.0, kT), raw=eT)
                REC.ident("D4d(iii).T_c: ratio multiplied by c^2", rel(ratio(pT), c * c * ratio(ps)))
                eR = rel(torch.stack(regs(gT["qi"], pT["P_t"]) + regs(gT["qu"], pT["P_u"])),
                         torch.stack(regs(g0["qi"], ps["P_t"]) + regs(g0["qu"], ps["P_u"])))
                REC.ident("D4d(iii).T_c: inclusion regularisers (full baskets) unchanged", eR / max(1.0, kT), raw=eR)
                mask = (rows.Om - rows.Omfull).abs().amax(1) > 0
                if mask.any():
                    REC.differ("D4d(iii).T_c: LOO training scores change", rel(oT["S"][mask], o0["S"][mask]))
        cg = 1e-2 * EPS / max(float(g0["qi"].norm(dim=1).max()), float(g0["qu"].norm(dim=1).max()), 1e-300)
        pc = dict(ps, E_w=cg * ps["E_w"], E_Iid=cg * ps["E_Iid"], E_Uid=cg * ps["E_Uid"],
                  W_t=ps["W_t"] / cg, W_u=ps["W_u"] / cg)
        if nz(g0["uh"]):
            REC.differ("D4c.guard regime (c*|q| < eps): S changes", rel(grid(pc, D, arm)["S"], g0["S"]))

    ka = ["D4a.sep per-side rescale, item side (E_I,W_t)", "D4a.sep per-side rescale, user side (E_U,W_u)"]
    sec.add("(a) exactness for c in [1e-3, 1e3]", V(REC.ok(*ka)), REC.s(*ka),
            "c in {1e-3, 0.37, 2.5, 1e3, lognormal(0,2)}; S, all four components and the LOO training scores. "
            "'exactly' = machine precision (c*E then W/c is not bitwise).")
    sec.add("(a) 'for any c > 0'", "FALSIFIED" if REC.ok("D4a.guard regime (c*|q_i| < eps): S changes") else "CONFIRMED",
            REC.s("D4a.guard regime (c*|q_i| < eps): S changes"),
            "With the spec's own guarded cosine (eps = 1e-8), once c*|q_i| < eps the cosine unit is no longer "
            "scale-invariant (cos -> |c q|cos/eps), so S changes. Correct statement: invariance holds for all c with "
            "c*min_i|q_i| >= eps (resp. c*min_u|q_u| >= eps on the user side); it fails below. Edge-of-definition "
            "failure, not a substantive one.")
    sec.add("(b)", V(REC.ok("D4b.shared partial rescale changes S", "D4b.shared partial rescale changes u^")),
            REC.s("D4b.shared partial rescale changes S", "D4b.shared partial rescale changes u^"))
    sec.add("(c) exactness for c in [1e-3, 1e3]", V(REC.ok("D4c.shared joint rescale: S, 4 components, LOO scores unchanged")),
            REC.s("D4c.shared joint rescale: S, 4 components, LOO scores unchanged"))
    sec.note("(c) does not say 'for any c', so the eps-guard edge is not counted against it; for the record the same "
             "edge applies: " + REC.s("D4c.guard regime (c*|q| < eps): S changes"))
    kd = ["D4d.shared joint rescale: ratio unchanged", "D4d.sep per-side: ratio' = ratio/c", "D4d.sep per-side: ratio changes"]
    sec.add("(d) invariance part", V(REC.ok(*kd)), REC.s(*kd))
    kd3 = ["D4d(iii).T_c: full-basket S + 4 components unchanged", "D4d(iii).T_c: ratio multiplied by c^2",
           "D4d(iii).T_c: inclusion regularisers (full baskets) unchanged", "D4d(iii).T_c: LOO training scores change"]
    sec.add("(d) parenthetical 'identified quantity under sharing'",
            "FALSIFIED" if REC.ok(*kd3[:3]) else "CONFIRMED", REC.s(*kd3),
            "Counter-map T_c in the shared IDS arm: E_w -> E_w/c, E_Iid -> E_Iid/c, W_t -> c W_t, W_u -> W_u/c, "
            "e_ID(u) -> c e_ID(u) + (c - 1/c) m_u with m_u = (1/|H_u|) sum_j sum_f e_f (full basket). Then q_i -> q_i/c, "
            "q_u(full) -> c q_u, so S, all four components and both regularisers are unchanged at every full-basket "
            "(inference) score while ||W_t||/||W_u|| is multiplied by c^2. So the ratio is NOT identified by the shared "
            "ids model's score function. What pins it is the training-time LOO: q_u^LOO(i) would need the per-user "
            "shift (c - 1/c) m_{u,-i}, which varies with i, so T_c changes the LOO training scores (generic data). "
            "Correct statement: 'identified under sharing' holds only for the noid arm or for the LOO training objective "
            "(and even there not when every basket's items have equal word sums, e.g. all |H_u| = 1 with the map "
            "e_ID(u) -> c e_ID(u)); for the ids arm's inference function it is a gauge quantity, as under sep. "
            "(Errors divided by kappa_T = max operand / |q_u'| of the map, since at c = 1e-3 the user-ID update "
            "cancels terms of size |m_u|/c.)")


# ============================================================== D5
def test_D5():
    sec = Section("D5", "Translation gauge of the ids arms -- fixed layout")
    adv_raw = adv_norm = 0.0
    adv_k = 0.0
    for s, sh, lay, arm, scn, rng in iter_cfg(layouts=("fixed",), arms=("ids",), tag=5):
        for table in ("shared", "sep"):
            D = Data(rng, sh, "fixed", basket=(2, 6))
            rows = Rows.sample(D, rng, 3)
            p, sc = make_params(rng, D, sh, table, scn)
            d = sh["d"]
            for vm in (1.0, 10.0, 1e6):
                v = blk(rng, (D.F, d), sc["E_w"] * vm)
                p2 = d5_map(p, D, v)
                k, big = kappa(p, p2, D, "ids", rows)
                kk = max(1.0, k)
                g1, g2 = grid(p, D, "ids"), grid(p2, D, "ids")
                o1, o2 = fwd_rows(p, D, "ids", rows), fwd_rows(p2, D, "ids", rows)
                errs = dict(
                    qi=rel(g1["qi"], g2["qi"], big), qu=rel(g1["qu"], g2["qu"], big), quloo=rel(o1["qu"], o2["qu"], big),
                    comps=max(rel(g1[c], g2[c]) for c in COMP_KEYS), loo=rel(o1["S"], o2["S"]),
                    regs=max(rel(torch.stack(regs(g1["qi"], p["P_t"]) + regs(g1["qu"], p["P_u"]) +
                                             regs(o1["qu"], p["P_u"]) + regs(o1["qc"].reshape(-1, d), p["P_t"])),
                                 torch.stack(regs(g2["qi"], p2["P_t"]) + regs(g2["qu"], p2["P_u"]) +
                                             regs(o2["qu"], p2["P_u"]) + regs(o2["qc"].reshape(-1, d), p2["P_t"]))), 0.0),
                    losses=max(rel(total(p, D, "ids", rows, kd, rg), total(p2, D, "ids", rows, kd, rg))
                               for kd in LOSSES for rg in (False, True)))
                if vm < 1e5:
                    REC.ident("D5a.q_i unchanged", errs["qi"])
                    REC.ident("D5a.q_u (full basket) unchanged", errs["qu"])
                    REC.ident("D5a.q_u^LOO (|H\\i|>=1) unchanged", errs["quloo"])
                    REC.ident("D5a.S + 4 components unchanged", errs["comps"] / kk, raw=errs["comps"])
                    REC.ident("D5a.LOO training scores unchanged", errs["loo"] / kk, raw=errs["loo"])
                    REC.ident("D5a.inclusion regularisers unchanged", errs["regs"] / kk, raw=errs["regs"])
                    REC.ident("D5a.three losses (+/- regs) unchanged", errs["losses"] / kk, raw=errs["losses"])
                else:
                    worst = max(errs.values())
                    adv_raw = max(adv_raw, worst)
                    adv_norm = max(adv_norm, max(errs["comps"], errs["loo"], errs["regs"], errs["losses"]) / kk,
                                   errs["qi"], errs["qu"], errs["quloo"])
                    adv_k = max(adv_k, k)
                    REC.ident("D5a.adversarial |v| = 1e6 |e|: kappa-normalised", max(
                        max(errs["comps"], errs["loo"], errs["regs"], errs["losses"]) / kk,
                        errs["qi"], errs["qu"], errs["quloo"]), raw=worst)
            # (b) F*d independent directions
            keys = sorted(p.keys())
            dirs = []
            g_l, gl = grads(p, lambda q: total(q, D, "ids", rows, "softmax", True))
            gref = flat(gl, keys)
            S0 = grid(p, D, "ids")["S"]
            O0 = fwd_rows(p, D, "ids", rows)["S"]
            for phi in range(D.F):
                for a in range(d):
                    vv = torch.zeros(D.F, d)
                    vv[phi, a] = 1.0
                    delta = flat(d5_map(p, D, vv), keys) - flat(p, keys)
                    dirs.append(delta)
                    t = float(rng.normal()) * sc["E_w"] * 3.0
                    p3 = d5_map(p, D, vv * t)
                    k3, _ = kappa(p, p3, D, "ids", rows)
                    e3 = max(rel(grid(p3, D, "ids")["S"], S0), rel(fwd_rows(p3, D, "ids", rows)["S"], O0),
                             rel(total(p3, D, "ids", rows, "bce", True), total(p, D, "ids", rows, "bce", True)))
                    REC.ident("D5b.each single-direction shift leaves S, LOO S, loss invariant", e3 / max(1.0, k3), raw=e3)
                    den = float((gref * delta).abs().sum())
                    if den > 0:
                        REC.ident("D5b.directional derivative of loss along each direction = 0",
                                  abs(float((gref * delta).sum())) / den)
            Dm = torch.stack(dirs).numpy()
            rk = np.linalg.matrix_rank(Dm)
            REC.check("D5b.rank of the F*d displacement set = F*d", rk == D.F * d)
            if table == "sep":
                v1 = blk(rng, (D.F, d), sc["E_w"])
                v2 = blk(rng, (D.F, d), sc["E_w"])
                p4 = d5_map(d5_map(p, D, v1, sides=("I",)), D, v2, sides=("U",))
                k4, _ = kappa(p, p4, D, "ids", rows)
                e4 = max(rel(grid(p4, D, "ids")["S"], S0), rel(fwd_rows(p4, D, "ids", rows)["S"], O0))
                REC.ident("D5(note).sep: independent v^I, v^U per table also invariant", e4 / max(1.0, k4), raw=e4)
                ext = list(dirs)
                for phi in range(D.F):
                    for a in range(d):
                        vv = torch.zeros(D.F, d)
                        vv[phi, a] = 1.0
                        ext.append(flat(d5_map(p, D, vv, sides=("I",)), keys) - flat(p, keys))
                REC.check("D5(note).sep: gauge translation dimension = 2*F*d",
                          np.linalg.matrix_rank(torch.stack(ext).numpy()) == 2 * D.F * d)
            # (c) |H_u| = 1 breakage
            Dc = Data(rng, sh, "fixed", basket=(1, 6))
            Dc.set_basket(0, [int(rng.integers(Dc.M))])
            pc, scc = make_params(rng, Dc, sh, table, scn)
            pool = [j for j in range(Dc.M) if j not in Dc.baskets[0]]
            rows1 = Rows(Dc, [(0, Dc.baskets[0][0], rng.choice(pool, size=3, replace=False).tolist())])
            v = blk(rng, (Dc.F, d), scc["E_w"])
            pc2 = d5_map(pc, Dc, v)
            o1, o2 = fwd_rows(pc, Dc, "ids", rows1), fwd_rows(pc2, Dc, "ids", rows1)
            _, bigc = kappa(pc, pc2, Dc, "ids")
            REC.ident("D5c.q_u^LOO(|H|=1) shifts by exactly -sum v", rel(o2["qu"] - o1["qu"], -v.sum(0)[None], bigc))
            REC.differ("D5c.LOO score of |H|=1 user changes", rel(o1["S"], o2["S"]))
    # (d) bag layout
    nusers_var = 0
    for s, sh, lay, arm, scn, rng in iter_cfg(layouts=("bag", "bagms"), arms=("ids",), tag=55):
        for table in ("shared", "sep"):
            D = Data(rng, sh, lay, basket=(2, 6))
            rows = Rows.sample(D, rng, 3)
            p, sc = make_params(rng, D, sh, table, scn)
            d = sh["d"]
            v = blk(rng, (D.F, d), sc["E_w"])
            p2 = d5_map(p, D, v)
            k, big = kappa(p, p2, D, "ids")
            q1 = comp_items(p, D, torch.arange(D.M), "ids")
            q2 = comp_items(p2, D, torch.arange(D.M), "ids")
            allone = (D.K == 1).all(1)
            for i in range(D.M):
                if allone[i]:
                    REC.ident("D5d.items with all k_{i,phi}=1 unchanged", rel(q1[i], q2[i], big))
                else:
                    REC.differ("D5d.items with some k_{i,phi}!=1 change", rel(q1[i], q2[i]))
            Kv = D.K @ v
            p3 = d5_map(p, D, v, item_shift=Kv)
            REC.ident("D5d.item-specific shift -sum_phi k_{i,phi} v_phi restores every q_i",
                      rel(comp_items(p3, D, torch.arange(D.M), "ids"), q1, big))
            Kbar_full = torch.stack([D.K[H].mean(0) for H in D.baskets])
            p4 = d5_map(p, D, v, item_shift=Kv, user_shift=Kbar_full @ v)
            g1, g4 = grid(p, D, "ids"), grid(p4, D, "ids")
            k4, _ = kappa(p, p4, D, "ids")
            e4 = max(rel(g1[c], g4[c]) for c in COMP_KEYS)
            REC.ident("D5d.inference exact with bag-count correction (item + per-user full-basket shift)", e4 / max(1.0, k4), raw=e4)
            o1, o4 = fwd_rows(p, D, "ids", rows), fwd_rows(p4, D, "ids", rows)
            for u, H in enumerate(D.baskets):
                req = [D.K[[j for j in H if j != i]].mean(0) @ v for i in H] + [Kbar_full[u] @ v]
                req = torch.stack(req)
                spread = _mx(req - req.mean(0)) / max(_mx(req), 1e-300)
                kvar = _mx(torch.stack([D.K[[j for j in H if j != i]].mean(0) for i in H]) - Kbar_full[u])
                if kvar > 0:
                    nusers_var += 1
                    REC.differ("D5d.required per-user shifts (full + every LOO) are not all equal", spread)
                    rr = [r for r, t in enumerate(rows.list) if t[0] == u]
                    REC.differ("D5d.LOO scores change under the full-basket per-user shift",
                               rel(o1["S"][rr], o4["S"][rr]))
                    # best single least-squares shift (mean of the required ones) also fails
                    bestu = Kbar_full @ v
                    bestu = bestu.clone()
                    bestu[u] = req.mean(0)
                    p5 = d5_map(p, D, v, item_shift=Kv, user_shift=bestu)
                    o5 = fwd_rows(p5, D, "ids", rows)
                    REC.differ("D5d.LOO scores change under the least-squares single per-user shift",
                               rel(o1["S"][rr], o5["S"][rr]))
    ka = ["D5a.q_i unchanged", "D5a.q_u (full basket) unchanged", "D5a.q_u^LOO (|H\\i|>=1) unchanged",
          "D5a.S + 4 components unchanged", "D5a.LOO training scores unchanged", "D5a.inclusion regularisers unchanged",
          "D5a.three losses (+/- regs) unchanged", "D5a.adversarial |v| = 1e6 |e|: kappa-normalised"]
    sec.add("(a)", V(REC.ok(*ka)), REC.s(*ka),
            "shared and sep (same v on both tables), |v| = 1x and 10x the word-row scale, plus an adversarial run with "
            "|v| = 1e6 x word scale. Composition errors are normalised by the largest embedding entry (operand scale); "
            "score/loss errors are divided by kappa = (largest embedding entry)/(smallest composition) -- the "
            "catastrophic-cancellation factor of (e+v) + (e_ID - sum v). Loss claim tested on data with all |H_u| >= 2 "
            "(as (c) shows, it is false otherwise).")
    kb = ["D5b.each single-direction shift leaves S, LOO S, loss invariant",
          "D5b.directional derivative of loss along each direction = 0", "D5b.rank of the F*d displacement set = F*d"]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb))
    kc = ["D5c.q_u^LOO(|H|=1) shifts by exactly -sum v", "D5c.LOO score of |H|=1 user changes"]
    sec.add("(c)", V(REC.ok(*kc)), REC.s(*kc))
    kd = ["D5d.items with all k_{i,phi}=1 unchanged", "D5d.items with some k_{i,phi}!=1 change",
          "D5d.item-specific shift -sum_phi k_{i,phi} v_phi restores every q_i",
          "D5d.inference exact with bag-count correction (item + per-user full-basket shift)",
          "D5d.required per-user shifts (full + every LOO) are not all equal",
          "D5d.LOO scores change under the full-basket per-user shift",
          "D5d.LOO scores change under the least-squares single per-user shift"]
    sec.add("(d)", V(REC.ok(*kd)), REC.s(*kd) + f"<br>users with varying per-field LOO bag mass: {nusers_var}",
            "'no single per-user shift' shown by (i) the required shifts {full, LOO(i) for i in H_u} being distinct "
            "vectors, and (ii) the LOO scores changing both for the full-basket shift and for the least-squares "
            "compromise shift.")
    sec.note("Observation (not a falsification of (b), which is about 'the map'): in the sep layout the item and user "
             "tables can be translated INDEPENDENTLY (v^I on E_I with item-ID compensation, v^U on E_U with user-ID "
             "compensation) and everything stays invariant, so the sep translation gauge is 2*F*d-dimensional; the "
             "shared layout ties the two and leaves F*d. " + REC.s("D5(note).sep: independent v^I, v^U per table also invariant",
                                                                   "D5(note).sep: gauge translation dimension = 2*F*d"))


# ============================================================== D6
def readouts(p, D, arm="ids"):
    Ewi, Eidi, _, _ = tabs(p, D.V)
    Wt, Pt = p["W_t"], p["P_t"]
    qi = comp_items(p, D, torch.arange(D.M), arm)
    nq = qi.norm(dim=1).clamp_min(EPS)
    nP = Pt.norm(dim=1).clamp_min(EPS)
    WI = torch.stack([ft[D.C[:, ft].argmax(1)] for ft in D.fields_t], 1)  # M x F word of item i in field phi
    r = dict(word_that=Ewi @ Wt.T, word_Bitem=Ewi @ Wt.sum(0), id_that=Eidi @ Wt.T, id_Bitem=Eidi @ Wt.sum(0),
             word_cosshare=(Ewi[WI] @ Pt.T) / nq[:, None, None] / nP, id_cosshare=(Eidi @ Pt.T) / nq[:, None] / nP,
             naming=cosm(Ewi, Pt))
    return r, WI


def centre(p, D, weighted=False):
    q = {k: x.clone() for k, x in p.items()}
    cnt = D.C.sum(0)

    def cw(Ew):
        ms = []
        for ft in D.fields_t:
            if weighted:
                w = cnt[ft]
                ms.append((w[:, None] * Ew[ft]).sum(0) / w.sum())
            else:
                ms.append(Ew[ft].mean(0))
        ms = torch.stack(ms)
        return Ew - ms[D.field_of], ms.sum(0)

    if "E_w" in p:
        Ew, sm = cw(p["E_w"])
        q["E_w"], q["E_Iid"], q["E_Uid"] = Ew, p["E_Iid"] + sm, p["E_Uid"] + sm
    else:
        Vn = D.V
        EI, smI = cw(p["E_I"][:Vn])
        EU, smU = cw(p["E_U"][:Vn])
        q["E_I"] = torch.cat([EI, p["E_I"][Vn:] + smI])
        q["E_U"] = torch.cat([EU, p["E_U"][Vn:] + smU])
    return q


def test_D6():
    sec = Section("D6", "Canonical gauge makes per-row read-outs invariant")
    nloo1 = 0
    for s, sh, lay, arm, scn, rng in iter_cfg(layouts=("fixed",), arms=("ids",), tag=6):
        for table in ("shared", "sep"):
            D = Data(rng, sh, "fixed", basket=(1, 6))
            rows = Rows.sample(D, rng, 3)
            p, sc = make_params(rng, D, sh, table, scn)
            Vn = D.V
            v = blk(rng, (D.F, sh["d"]), sc["E_w"])
            p2 = d5_map(p, D, v)
            k, big = kappa(p, p2, D, "ids")
            r1, WI = readouts(p, D)
            r2, _ = readouts(p2, D)
            for key in RO_KEYS:
                REC.differ(f"D6a.uncentred {key} changes", rel(r1[key], r2[key]))
            c1, c2 = centre(p, D), centre(p2, D)
            ek = emb_keys(p)
            REC.ident("D6b.centred rows identical (theta vs theta')", max(rel(c1[e], c2[e], big) for e in ek))
            kc = big / max(min(float(c1[e].abs().amax(1).min()) for e in ek), 1e-300)
            rc1, _ = readouts(c1, D)
            rc2, _ = readouts(c2, D)
            ero = max(rel(rc1[key], rc2[key]) for key in RO_KEYS)
            REC.ident("D6b.centred read-outs identical", ero / max(1.0, kc), raw=ero)
            # (c)
            gc, g = grid(c1, D, "ids"), grid(p, D, "ids")
            REC.ident("D6c.sum centred word rows + centred ID = q_i", rel(gc["qi"], g["qi"], big))
            ec = max(rel(gc[kk], g[kk]) for kk in COMP_KEYS)
            REC.ident("D6c.S + 4 components identical from centred rows (full baskets)", ec / max(1.0, k), raw=ec)
            ws_t = rc1["word_that"][WI]
            REC.ident("D6c.sum centred t^ shares = t^", rel(ws_t.sum(1) + rc1["id_that"], g["th"],
                                                             _mx(ws_t.abs().sum(1) + rc1["id_that"].abs())))
            ws_b = rc1["word_Bitem"][WI]
            REC.ident("D6c.sum centred B_item shares = B_item", rel(ws_b.sum(1) + rc1["id_Bitem"], g["th"].sum(1),
                                                                    _mx(ws_b.abs().sum(1) + rc1["id_Bitem"].abs())))
            REC.ident("D6c.(extra) sum centred cosine-unit shares = cos(q_i,P_t)",
                      rel(rc1["word_cosshare"].sum(1) + rc1["id_cosshare"], g["ts"] - 1.0,
                          _mx(rc1["word_cosshare"].abs().sum(1) + rc1["id_cosshare"].abs())))
            single = [r for r, t in enumerate(rows.list) if len(D.baskets[t[0]]) == 1]
            if single:
                nloo1 += 1
                oc, o = fwd_rows(c1, D, "ids", rows), fwd_rows(p, D, "ids", rows)
                REC.differ("D6c.caveat: LOO score of |H_u|=1 rows changes under centring", rel(oc["S"][single], o["S"][single]))
            # (d)
            E1, E2 = tabs(p, Vn)[0], tabs(p2, Vn)[0]
            pairs = [(int(a), int(b)) for ws in D.fields for a in ws for b in ws if a < b]
            fa = torch.tensor([x[0] for x in pairs])
            fb = torch.tensor([x[1] for x in pairs])
            con1, con2 = E1[fb] - E1[fa], E2[fb] - E2[fa]
            REC.ident("D6d.within-field contrasts e_f' - e_f invariant", rel(con1, con2, big))
            kcon = big / max(float(con1.abs().amax(1).min()), 1e-300)
            sub1 = g["us"] @ p["W_t"] @ con1.T
            sub2 = grid(p2, D, "ids")["us"] @ p2["W_t"] @ con2.T
            es = rel(sub1, sub2)
            REC.ident("D6d.substitution counterfactual u*^T W_t (e_f' - e_f) invariant", es / max(1.0, kcon, k), raw=es)
            rem1 = g["us"] @ p["W_t"] @ E1.T
            rem2 = grid(p2, D, "ids")["us"] @ p2["W_t"] @ E2.T
            REC.differ("D6d.removal effect u*^T W_t e_f changes", rel(rem1, rem2))
            # (e)
            w1, w2 = centre(p, D, True), centre(p2, D, True)
            REC.ident("D6e.frequency-weighted centring invariant under D5", max(rel(w1[e], w2[e], big) for e in ek))
            REC.differ("D6e.frequency-weighted rows != canonical rows", max(rel(w1[e], c1[e]) for e in ek))
    sec.add("(a)", V(REC.ok(*[f"D6a.uncentred {k} changes" for k in RO_KEYS])),
            REC.s(*[f"D6a.uncentred {k} changes" for k in RO_KEYS]),
            "word_that = W_t e_f, word_Bitem = (W_t^T 1).e_f, id_* = the ID-row shares of t^, B_item and the cosine "
            "unit, word_cosshare = e_f.P_t[k]/(|q_i||P_t[k]|), naming = cos(e_f, P_t[k]). shared and sep (sep "
            "centred per table with its own field means; user IDs take the user table's means).")
    sec.add("(b)", V(REC.ok("D6b.centred rows identical (theta vs theta')", "D6b.centred read-outs identical")),
            REC.s("D6b.centred rows identical (theta vs theta')", "D6b.centred read-outs identical"))
    kc = ["D6c.sum centred word rows + centred ID = q_i", "D6c.S + 4 components identical from centred rows (full baskets)",
          "D6c.sum centred t^ shares = t^", "D6c.sum centred B_item shares = B_item",
          "D6c.(extra) sum centred cosine-unit shares = cos(q_i,P_t)"]
    sec.add("(c)", V(REC.ok(*kc)), REC.s(*kc),
            "S read as the inference score (full baskets, as defined in the mechanism). Caveat measured below: the "
            "canonical centring IS a D5 map (v_phi = -mean_phi), so it inherits D5(c): the training-time LOO score of a "
            "|H_u| = 1 user is NOT preserved by centring.")
    sec.note("D6(c) caveat: " + REC.s("D6c.caveat: LOO score of |H_u|=1 rows changes under centring") +
             f" (configs with |H_u|=1 rows: {nloo1}). Not counted against (c) because S/components are inference-time "
             "quantities; it matters if centred rows are ever fed back into LOO training.")
    kd = ["D6d.within-field contrasts e_f' - e_f invariant", "D6d.substitution counterfactual u*^T W_t (e_f' - e_f) invariant",
          "D6d.removal effect u*^T W_t e_f changes"]
    sec.add("(d)", V(REC.ok(*kd)), REC.s(*kd))
    ke = ["D6e.frequency-weighted centring invariant under D5", "D6e.frequency-weighted rows != canonical rows"]
    sec.add("(e)", V(REC.ok(*ke)), REC.s(*ke), "weights = word counts over the item catalogue (column sums of the item-word count matrix).")


# ============================================================== D7
def test_D7():
    sec = Section("D7", "Cold conventions under the gauge")
    fl_cc = n_cc = fl_cw = n_cw = 0
    ex_cc = ex_cw = None
    for s, sh, lay, arm, scn, rng in iter_cfg(layouts=("fixed",), arms=("ids",), tag=7):
        for table in ("shared", "sep"):
            D = Data(rng, sh, "fixed", basket=(1, 6))
            p, sc = make_params(rng, D, sh, table, scn)
            Vn, d = D.V, sh["d"]
            Mc = 8
            Cc = torch.zeros(Mc, Vn)
            for c in range(Mc):
                for ws in D.fields:
                    Cc[c, int(rng.choice(ws))] = 1.0
            Bc = [sorted(rng.choice(D.M, size=int(rng.integers(1, 5)), replace=False).tolist()) for _ in range(5)]
            Omc = torch.stack([D.omega(b) for b in Bc])
            for vm in (1.0, 3.0):
                v = blk(rng, (D.F, d), sc["E_w"] * vm)
                sv = v.sum(0)
                p2 = d5_map(p, D, v)
                k, big = kappa(p, p2, D, "ids")
                kk = max(1.0, k)
                T1, T2 = tabs(p, Vn), tabs(p2, Vn)
                g1, g2 = grid(p, D, "ids"), grid(p2, D, "ids")
                qc1, qc2 = Cc @ T1[0], Cc @ T2[0]
                REC.ident("D7a.drop: q_cold' - q_cold = sum_phi v_phi", rel(qc2 - qc1, sv.expand_as(qc1), big))
                ts1, th1 = iroles(p, qc1)
                ts2, th2 = iroles(p2, qc2)
                REC.differ("D7a.drop: cold t* differs", rel(ts1, ts2))
                REC.differ("D7a.drop: cold t^ differs", rel(th1, th2))
                Sc1 = g1["us"] @ th1.T + g1["uh"] @ ts1.T
                Sc2 = g2["us"] @ th2.T + g2["uh"] @ ts2.T
                REC.differ("D7a.drop: S(u,cold) differs", rel(Sc1, Sc2))
                L1, L2 = g1["us"] @ th1.T, g2["us"] @ th2.T
                dl1, dl2 = L1[:, :, None] - L1[:, None, :], L2[:, :, None] - L2[:, None, :]
                el = rel(dl1, dl2, _mx(L1), _mx(L2))
                REC.ident("D7a.drop: cold-pair difference of the linear part u*^T t^ is invariant", el / kk, raw=el)
                sc_ = max(_mx(Sc1), _mx(Sc2))
                d1, d2 = Sc1[:, :, None] - Sc1[:, None, :], Sc2[:, :, None] - Sc2[:, None, :]
                m = (torch.triu(torch.ones(Mc, Mc, dtype=torch.bool), 1)[None].expand_as(d1)
                     & (d1.abs() > 1e-8 * sc_) & (d2.abs() > 1e-8 * sc_))
                fl = (torch.sign(d1) != torch.sign(d2)) & m
                fl_cc += int(fl.sum())
                n_cc += int(m.sum())
                if ex_cc is None and fl.any():
                    u, a, b = [int(x) for x in torch.nonzero(fl)[0]]
                    ex_cc = (f"seed {s}, shape {sh['name']}, {table}, scale {scn}: user {u}, cold items {a},{b}: "
                             f"S_theta = ({float(Sc1[u, a]):.6g}, {float(Sc1[u, b]):.6g}), "
                             f"S_theta' = ({float(Sc2[u, a]):.6g}, {float(Sc2[u, b]):.6g}); linear-part difference "
                             f"{float(dl1[u, a, b]):.6g} vs {float(dl2[u, a, b]):.6g} (equal), so the flip is through t*")
                e1 = Sc1[:, :, None] - g1["S"][:, None, :]
                e2 = Sc2[:, :, None] - g2["S"][:, None, :]
                sw = max(sc_, _mx(g1["S"]))
                m2 = (e1.abs() > 1e-8 * sw) & (e2.abs() > 1e-8 * sw)
                fl2 = (torch.sign(e1) != torch.sign(e2)) & m2
                fl_cw += int(fl2.sum())
                n_cw += int(m2.sum())
                if ex_cw is None and fl2.any():
                    u, a, w = [int(x) for x in torch.nonzero(fl2)[0]]
                    ex_cw = (f"seed {s}, shape {sh['name']}, {table}, scale {scn}: user {u}, cold {a} vs warm {w}: "
                             f"S_theta = ({float(Sc1[u, a]):.6g} vs {float(g1['S'][u, w]):.6g}), "
                             f"S_theta' = ({float(Sc2[u, a]):.6g} vs {float(g2['S'][u, w]):.6g})")
                REC.ident("D7(aux).warm scores invariant", rel(g1["S"], g2["S"]) / kk)
                # (b)
                eb1, eb2 = T1[1].mean(0), T2[1].mean(0)
                REC.ident("D7b.mean-ID shifts by exactly -sum v", rel(eb2 - eb1, -sv, big))
                qm1, qm2 = qc1 + eb1, qc2 + eb2
                REC.ident("D7b.imputed cold composition identical", rel(qm1, qm2, big))
                tsm1, thm1 = iroles(p, qm1)
                tsm2, thm2 = iroles(p2, qm2)
                Sm1 = g1["us"] @ thm1.T + g1["uh"] @ tsm1.T
                Sm2 = g2["us"] @ thm2.T + g2["uh"] @ tsm2.T
                km = big / max(float(qm1.abs().amax(1).min()), 1e-300)
                em = rel(Sm1, Sm2)
                REC.ident("D7b.imputed cold scores agree", em / max(1.0, km, k), raw=em)
                # (c)
                qd1, qd2 = Omc @ T1[2], Omc @ T2[2]
                ebu1, ebu2 = T1[3].mean(0), T2[3].mean(0)
                REC.differ("D7c.cold user (non-empty basket), drop: scores differ",
                           rel(grid(p, D, "ids", qu=qd1)["S"], grid(p2, D, "ids", qu=qd2)["S"]))
                ki = big / max(float((qd1 + ebu1).abs().amax(1).min()), 1e-300)
                ei = rel(grid(p, D, "ids", qu=qd1 + ebu1)["S"], grid(p2, D, "ids", qu=qd2 + ebu2)["S"])
                REC.ident("D7c.cold user (non-empty basket), impute mean user-ID: scores agree", ei / max(1.0, ki, k), raw=ei)
                z = torch.zeros(1, d)
                REC.ident("D7c.empty basket, q_u = 0: scores agree", rel(grid(p, D, "ids", qu=z)["S"], grid(p2, D, "ids", qu=z)["S"]) / kk)
                REC.differ("D7c.empty basket, impute mean user-ID: scores differ",
                           rel(grid(p, D, "ids", qu=ebu1[None])["S"], grid(p2, D, "ids", qu=ebu2[None])["S"]))
    REC.check("D7a.found a cold-cold ranking flip (cosine unit)", fl_cc > 0)
    REC.check("D7a.found a cold-warm ranking flip", fl_cw > 0)
    ka = ["D7a.drop: q_cold' - q_cold = sum_phi v_phi", "D7a.drop: cold t* differs", "D7a.drop: cold t^ differs",
          "D7a.drop: S(u,cold) differs", "D7a.drop: cold-pair difference of the linear part u*^T t^ is invariant",
          "D7a.found a cold-cold ranking flip (cosine unit)", "D7a.found a cold-warm ranking flip", "D7(aux).warm scores invariant"]
    sec.add("(a)", V(REC.ok(*ka)), REC.s(*ka) +
            f"<br>cold-cold pairs flipped: {fl_cc} / {n_cc}; cold-warm pairs flipped: {fl_cw} / {n_cw}"
            f"<br>example cold-cold: {ex_cc}<br>example cold-warm: {ex_cw}",
            "for a fixed user the drop-convention shift adds the SAME constant u*^T W_t sum v to every cold item's linear "
            "part, so cold-cold flips can only come through the cosine unit -- verified (the linear-part pair "
            "difference is invariant).")
    kb = ["D7b.mean-ID shifts by exactly -sum v", "D7b.imputed cold composition identical", "D7b.imputed cold scores agree"]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb))
    kc = ["D7c.cold user (non-empty basket), drop: scores differ",
          "D7c.cold user (non-empty basket), impute mean user-ID: scores agree",
          "D7c.empty basket, q_u = 0: scores agree", "D7c.empty basket, impute mean user-ID: scores differ"]
    sec.add("(c)", V(REC.ok(*kc)), REC.s(*kc))


# ============================================================== D8
def twoway(Hm):
    g = Hm.mean()
    r = Hm.mean(1, keepdim=True) - g
    c = Hm.mean(0, keepdim=True) - g
    it = Hm - g - r - c
    tot = float(((Hm - g) ** 2).sum())
    ssr = float((r ** 2).sum()) * Hm.shape[1]
    ssc = float((c ** 2).sum()) * Hm.shape[0]
    ssi = float((it ** 2).sum())
    if tot == 0:
        return 0.0, 0.0, 0.0, 0.0
    return ssc / tot, ssr / tot, ssi / tot, tot / Hm.numel()


def test_D8():
    sec = Section("D8", "Common-mode mass in the personalised halves")
    sigmas = (1.0, 0.3, 0.1, 0.03, 1e-2, 1e-3, 1e-4, 1e-6)
    readings = (("R1: ids, e_ID(u) per-user, fixed scale tau=0.3", "fixed", 0.3),
                ("R1': ids, e_ID(u) per-user, fixed scale tau=0.05", "fixed", 0.05),
                ("R2: ids, e_ID(u) scaled with sigma (tau=0.3*sigma)", "scaled", 0.3),
                ("(c): noid, e_ID(u)=0", "zero", 0.0))
    tab = {rd[0]: {sg: dict(U=[], I=[], Uitem=[], Iitem=[], Uvar=[], Ivar=[]) for sg in sigmas} for rd in readings}
    rank_users = []
    for seed in SEEDS:
        rng = np.random.default_rng([8, seed])
        N, M, d, Ku, Kt = 40, 30, 8, 5, 6
        p = dict(P_u=torch.tensor(rng.normal(size=(Ku, d))), P_t=torch.tensor(rng.normal(size=(Kt, d))),
                 W_u=torch.tensor(rng.normal(size=(Kt, d))), W_t=torch.tensor(rng.normal(size=(Ku, d))))
        mu = torch.tensor(rng.normal(size=d))
        Qi = torch.tensor(rng.normal(size=(M, d)))
        Z = torch.tensor(rng.normal(size=(N, d)))
        Y = torch.tensor(rng.normal(size=(N, d)))
        ts, th = iroles(p, Qi)
        musim, muhat = uroles(p, mu[None])
        I_cm = (muhat @ (ts - 1.0).T)[0]       # (W_u mu)^T (t*(i) - 1)
        U_cm = ((musim - 1.0) @ th.T)[0]       # (sim(mu,P_u) - 1)^T t^(i)
        for name, mode, tau in readings:
            for sg in sigmas:
                eid = tau * Y if mode == "fixed" else (tau * sg * Y if mode == "scaled" else torch.zeros(N, d))
                qu = mu + sg * Z + eid
                us, uh = uroles(p, qu)
                Uh = (us - 1.0) @ th.T
                Ih = uh @ (ts - 1.0).T
                cu = twoway(Uh)
                ci = twoway(Ih)
                t = tab[name][sg]
                t["U"].append(cu[1] + cu[2])
                t["I"].append(ci[1] + ci[2])
                t["Uitem"].append(cu[0])
                t["Iitem"].append(ci[0])
                t["Uvar"].append(cu[3])
                t["Ivar"].append(ci[3])
                if sg == 1.0:
                    # exact per-item split of the I-half
                    REC.ident("D8a.I-half = (W_u mu)^T(t*-1) + per-user remainder (exact)",
                              rel(Ih - I_cm[None], ((sg * Z + eid) @ p["W_u"].T) @ (ts - 1.0).T, _mx(Ih)))
                if sg in (1.0, 0.1) and mode in ("fixed", "zero"):
                    S = us @ th.T + uh @ ts.T
                    ch = 0
                    for Smod in (S - I_cm[None] - U_cm[None], S - I_cm[None], S - U_cm[None]):
                        ch = max(ch, sum(int(not torch.equal(torch.argsort(S[u]), torch.argsort(Smod[u]))) for u in range(N)))
                    rank_users.append(ch)
                    REC.check("D8a.common-mode terms are rank-active (>=1 user's ranking changes)", ch > 0)
        REC.check("D8a.common-mode terms are per-item (identical across users)", True)  # by construction
    # verdicts
    def lim(name, key):
        return max(tab[name][sigmas[-1]][key])

    def minitem(name, key):
        return min(tab[name][sigmas[-1]][key])

    def varratio(name, key):
        return min(np.array(tab[name][sigmas[-1]][key]) / np.array(tab[name][sigmas[0]][key]))

    r1 = readings[0][0]
    r1b = readings[1][0]
    r2 = readings[2][0]
    rc = readings[3][0]
    r1_lim = min(lim(r1, "U"), lim(r1, "I"))
    r2_ok = max(lim(r2, "U"), lim(r2, "I")) < 1e-8 and min(minitem(r2, "Uitem"), minitem(r2, "Iitem")) > 1 - 1e-8
    rc_ok = max(lim(rc, "U"), lim(rc, "I")) < 1e-8 and min(minitem(rc, "Uitem"), minitem(rc, "Iitem")) > 1 - 1e-8
    var_ok = min(varratio(n, k) for n in (r2, rc) for k in ("Uvar", "Ivar")) > 1e-2
    sec.add("(a)", V(REC.ok("D8a.I-half = (W_u mu)^T(t*-1) + per-user remainder (exact)",
                            "D8a.common-mode terms are rank-active (>=1 user's ranking changes)")),
            REC.s("D8a.I-half = (W_u mu)^T(t*-1) + per-user remainder (exact)",
                  "D8a.common-mode terms are rank-active (>=1 user's ranking changes)") +
            f"<br>users (of 40) whose ranking changes when the common-mode term(s) are removed, per instance: "
            f"min {min(rank_users)}, max {max(rank_users)}",
            "both terms are functions of i only by construction; rank-activity tested by re-ranking each user on "
            "S - I_cm - U_cm, S - I_cm and S - U_cm.")
    lines = ["| sigma | " + " | ".join(f"{n.split(':')[0]} U-half user+int / I-half user+int" for n, _, _ in readings) + " |",
             "|---|" + "---|" * len(readings)]
    for sg in sigmas:
        lines.append(f"| {sg:g} | " + " | ".join(f"{max(tab[n][sg]['U']):.2e} / {max(tab[n][sg]['I']):.2e}" for n, _, _ in readings) + " |")
    table_md = "\n".join(lines)
    sec.add("(b) literal reading (ids arm, e_ID(u) per-user of fixed scale, only delta_u -> 0)",
            "FALSIFIED" if r1_lim > 1e-3 else "CONFIRMED",
            f"limit (sigma=1e-6) user-main + interaction share, min over halves of max over seeds: R1 (tau=0.3) "
            f"{r1_lim:.3f}; R1' (tau=0.05) {min(lim(r1b, 'U'), lim(r1b, 'I')):.3e}. Item-main share at sigma=1e-6 under "
            f"R1: U-half {minitem(r1, 'Uitem'):.3f}, I-half {minitem(r1, 'Iitem'):.3f} (does not tend to 1).",
            "The construction q_u = mu + delta_u + e_ID(u) keeps the per-user ID rows when sigma = scale(delta_u) -> 0, so "
            "users still differ by e_ID(u) and the user-main/interaction shares converge to the share produced by the "
            "e_ID(u) spread alone -- nonzero, decreasing with tau but not vanishing. 'Zero personalisation' is not "
            "reached by sigma -> 0 in the ids arm. Correct statement: the shares -> 0 (item share -> 1) iff the WHOLE "
            "per-user part delta_u + e_ID(u) -> 0 (or e_ID(u) is common to all users, i.e. absorbed in mu).")
    sec.add("(b) with the per-user part delta_u + e_ID(u) of scale sigma (R2)", V(r2_ok and var_ok),
            f"R2 limit user+int share max {max(lim(r2, 'U'), lim(r2, 'I')):.2e}; item share min "
            f"{min(minitem(r2, 'Uitem'), minitem(r2, 'Iitem')):.10f}; total-variance ratio var(sigma=1e-6)/var(sigma=1) "
            f"min {min(varratio(r2, 'Uvar'), varratio(r2, 'Ivar')):.3f} (does not tend to 0)")
    sec.add("(c)", V(rc_ok and var_ok),
            f"noid limit user+int share max {max(lim(rc, 'U'), lim(rc, 'I')):.2e}; item share min "
            f"{min(minitem(rc, 'Uitem'), minitem(rc, 'Iitem')):.10f}; total-variance ratio min "
            f"{min(varratio(rc, 'Uvar'), varratio(rc, 'Ivar')):.3f}")
    sec.note("sigma sweep (max over 6 seeds of the user-main + interaction variance share; N=40 users, M=30 items, d=8, "
             "K_u=5, K_t=6, |mu| ~ N(0,1) entries, items N(0,1)):\n\n" + table_md)


# ============================================================== D9
def AB(p):
    Pu = p["P_u"] / p["P_u"].norm(dim=1, keepdim=True).clamp_min(EPS)
    Pt = p["P_t"] / p["P_t"].norm(dim=1, keepdim=True).clamp_min(EPS)
    return Pu.T @ p["W_t"], Pt.T @ p["W_u"]


def pair_terms(p, D, arm, u, i, Mui, om_u):
    Ewi, Eidi, Ewu, Eidu = tabs(p, D.V)
    fu = torch.nonzero(om_u).flatten()
    wu, Ru = om_u[fu], Ewu[fu]
    gi = torch.nonzero(D.C[i]).flatten()
    wi, Ri = D.C[i][gi], Ewi[gi]
    if arm == "ids":
        Ru = torch.cat([Ru, Eidu[u][None]])
        wu = torch.cat([wu, torch.ones(1)])
        Ri = torch.cat([Ri, Eidi[i][None]])
        wi = torch.cat([wi, torch.ones(1)])
    return wu[:, None] * (Ru @ Mui @ Ri.T) * wi[None, :]


def pers_scaled(p, D, arm, u, i, f, c):
    mask = torch.zeros(D.V, 1)
    mask[f] = 1.0
    q = dict(p, E_w=p["E_w"] * (1.0 + (c - 1.0) * mask))
    qu = comp_users(q, D, [u], D.om_full()[u][None], arm)
    qi = comp_items(q, D, [i], arm)
    us, uh = uroles(q, qu)
    ts, th = iroles(q, qi)
    return ((us - 1.0) * th).sum() + (uh * (ts - 1.0)).sum()


def test_D9():
    sec = Section("D9", "Overlap / self-affinity channel")
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=9, seeds=SEEDS[:3]):
        for table in ("shared", "sep"):
            D = Data(rng, sh, lay)
            p, sc = make_params(rng, D, sh, table, scn)
            g = grid(p, D, arm)
            pers = g["U_half"] + g["I_half"]
            A, B = AB(p)
            nqu = g["qu"].norm(dim=1).clamp_min(EPS)
            nqi = g["qi"].norm(dim=1).clamp_min(EPS)
            Om = D.om_full()
            for _ in range(12):
                u, i = int(rng.integers(D.N)), int(rng.integers(D.M))
                Mui = A / nqu[u] + B.T / nqi[i]
                bil = g["qu"][u] @ Mui @ g["qi"][i]
                mag = float((g["us"][u] - 1.0).abs() @ g["th"][i].abs() + g["uh"][u].abs() @ (g["ts"][i] - 1.0).abs())
                REC.ident("D9a.pers = q_u^T M_ui q_i", abs(float(bil - pers[u, i])) / max(mag, abs(float(bil)), 1e-300))
                T = pair_terms(p, D, arm, u, i, Mui, Om[u])
                REC.ident("D9a.pair-sum reproduces pers", abs(float(T.sum() - pers[u, i])) /
                          max(mag, float(T.abs().sum()), 1e-300))
    # (b)
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=92, seeds=SEEDS[:3]):
        D = Data(rng, sh, lay)
        ps, _ = make_params(rng, D, sh, "shared", scn)
        pu_, _ = make_params(rng, D, sh, "sep", scn)
        Om = D.om_full()
        hit = None
        for u in range(D.N):
            for i in range(D.M):
                both = torch.nonzero((Om[u] > 0) & (D.C[i] > 0)).flatten()
                if both.numel():
                    hit = (u, i, int(both[0]))
                    break
            if hit:
                break
        if hit is None:
            continue
        u, i, f = hit
        for pp, tab_ in ((ps, "shared"), (pu_, "sep")):
            g = grid(pp, D, arm)
            A, B = AB(pp)
            Mui = A / g["qu"][u].norm().clamp_min(EPS) + B.T / g["qi"][i].norm().clamp_min(EPS)
            w, n = float(Om[u, f]), float(D.C[i, f])
            Ewi, _, Ewu, _ = tabs(pp, D.V)
            eI, eU = Ewi[f], Ewu[f]
            qu, qi = g["qu"][u], g["qi"][i]
            for c in (1e-3, 0.4, 2.7, 1e3):
                if tab_ == "shared":
                    diag = lambda cc: w * n * (cc * eI) @ Mui @ (cc * eI)
                    REC.ident("D9b.shared: diagonal term scales by c^2", rel(diag(c), c * c * diag(1.0)))
                    pf = lambda cc: (qu + w * (cc - 1.0) * eU) @ Mui @ (qi + n * (cc - 1.0) * eI)
                    # exact quadratic in x = c-1: k0 + x (w e^T M q_i + n q_u^T M e) + x^2 w n e^T M e
                    x = c - 1.0
                    k0 = qu @ Mui @ qi
                    k1 = w * (eU @ Mui @ qi) + n * (qu @ Mui @ eI)
                    k2 = w * n * (eI @ Mui @ eI)
                    REC.ident("D9b.shared: frozen-M pers is exactly quadratic in c, x^2-coefficient w_uf n_if e_f^T M e_f",
                              rel(pf(c), k0 + k1 * x + k2 * x * x, abs(float(k0)) + abs(float(k1 * x)) + abs(float(k2 * x * x))))
                else:
                    bl = lambda cc: w * n * eU @ Mui @ (cc * eI)
                    REC.ident("D9b.sep: bilinear term (scale e^I_f only) scales by c", rel(bl(c), c * bl(1.0)))
                    pf = lambda cc: qu @ Mui @ (qi + n * (cc - 1.0) * eI)
                    f0, f1 = pf(0.0), pf(1.0)
                    REC.ident("D9b.sep: frozen-M pers is exactly linear in c (e^I_f only)",
                              rel(pf(c), f0 + (f1 - f0) * c, abs(float(f0)) + abs(float((f1 - f0) * c))))
    # (c)
    npos = nneg = 0
    ex = []
    for seed in SEEDS:
        rng = np.random.default_rng([9, 3, seed])
        sh = dict(name="D9c", sizes=(4, 5, 3), M=10, N=5, d=6, Ku=3, Kt=4)
        D = Data(rng, sh, "fixed", basket=(2, 4))
        for arm in ("ids", "noid"):
            p, _ = make_params(rng, D, sh, "shared", "unit")
            u = 0
            i = D.baskets[0][0]
            f = D.words(i)[0]
            cs = (1e2, 1e3, 1e4, 1e5, 1e6)
            rat = [float(pers_scaled(p, D, arm, u, i, f, torch.tensor(c))) / c for c in cs]
            conv = abs(rat[-1] - rat[-2]) / max(abs(rat[-1]), 1e-300)
            c1 = torch.tensor(1.0, requires_grad=True)
            y = pers_scaled(p, D, arm, u, i, f, c1)
            (g1,) = torch.autograd.grad(y, c1, create_graph=True)
            (g2,) = torch.autograd.grad(g1, c1)
            REC.check("D9c.not polynomial in c (finite lim pers/c but pers''(1) != 0)",
                      conv < 1e-3 and abs(float(g2)) > 1e-6 * (abs(float(y)) + abs(float(g1))))
            # M_ui-aligned shared row
            for sign in (+1, -1):
                Ew = p["E_w"].clone()
                alpha = 5.0 * float(p["E_w"].norm(dim=1).max())
                for _ in range(6):
                    q = dict(p, E_w=Ew)
                    g = grid(q, D, arm)
                    A, B = AB(q)
                    Mui = A / g["qu"][u].norm() + B.T / g["qi"][i].norm()
                    ev, evec = torch.linalg.eigh(0.5 * (Mui + Mui.T))
                    vec = evec[:, -1] if sign > 0 else evec[:, 0]
                    Ew = Ew.clone()
                    Ew[f] = alpha * vec
                q = dict(p, E_w=Ew)
                c1 = torch.tensor(1.0, requires_grad=True)
                y = pers_scaled(q, D, arm, u, i, f, c1)
                (gd,) = torch.autograd.grad(y, c1)
                if sign > 0:
                    npos += int(float(gd) > 0)
                    if len(ex) < 2 and float(gd) > 0:
                        ex.append(f"seed {seed} {arm}: dpers/dc|_{{c=1}} = {float(gd):+.4g} (pers = {float(y):.4g})")
                else:
                    nneg += int(float(gd) < 0)
    ka = ["D9a.pers = q_u^T M_ui q_i", "D9a.pair-sum reproduces pers"]
    sec.add("(a)", V(REC.ok(*ka)), REC.s(*ka),
            "shared and sep, fixed and bag, ids and noid, 12 random (u,i) pairs per config; M_ui uses the guarded "
            "norms max(|q|, eps), which is what the cosine uses; error normalised by sum |(u*-1)_l t^_l| + "
            "sum |u^_k (t*-1)_k| (and by sum |pair terms| for the pair-sum).")
    kb = ["D9b.shared: diagonal term scales by c^2",
          "D9b.shared: frozen-M pers is exactly quadratic in c, x^2-coefficient w_uf n_if e_f^T M e_f",
          "D9b.sep: bilinear term (scale e^I_f only) scales by c",
          "D9b.sep: frozen-M pers is exactly linear in c (e^I_f only)"]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb),
            "in the bag layout with repeated words the diagonal term carries the candidate multiplicity n_{i,f}: "
            "w_{u,f} n_{i,f} e_f^T M e_f (the spec's formula is the n=1 case).")
    sec.add("(c)", V(REC.ok("D9c.not polynomial in c (finite lim pers/c but pers''(1) != 0)") and npos > 0),
            REC.s("D9c.not polynomial in c (finite lim pers/c but pers''(1) != 0)") +
            f"<br>M_ui-aligned constructions with dpers/dc > 0 at c=1: {npos}/12; anti-aligned (bottom eigenvector) "
            f"with dpers/dc < 0: {nneg}/12<br>examples: " + "; ".join(ex),
            "non-polynomiality argued as: pers(c)/c converges to a finite limit as c -> inf (so a polynomial would have "
            "degree <= 1) while pers''(1) != 0. The positive-sign case uses e_f = alpha * top eigenvector of "
            "sym(M_ui), iterated to self-consistency.")


# ============================================================== D10
def ls_alpha(Q, t, rtol=1e-9):
    alpha = torch.linalg.pinv(Q.T, rtol=rtol) @ t
    res = float((Q.T @ alpha - t).norm()) / max(float(t.norm()), 1e-300)
    return alpha, res


def test_D10():
    sec = Section("D10", "Nesting of the user cloud")
    ndep = nmis = 0
    nind = 0
    for s, sh, lay, arm, scn, rng in iter_cfg(arms=("noid", "ids"), tag=10):
        D = Data(rng, sh, lay)
        p, _ = make_params(rng, D, sh, "shared", scn)
        g = grid(p, D, arm)
        _, Eidi, _, Eidu = tabs(p, D.V)
        for u, H in enumerate(D.baskets):
            Q = g["qi"][H] - (Eidi[H] if arm == "ids" else 0.0)
            t = g["qu"][u] - (Eidu[u] if arm == "ids" else 0.0)
            idm = max(_mx(Eidi[H]), _mx(Eidu[u])) if arm == "ids" else 0.0
            REC.ident(f"D10.convex identity q_u(-e_ID) = mean_j q_j(-e_ID) ({arm})", rel(t, Q.mean(0), idm))
            if not nz(t):
                continue
            alpha, res = ls_alpha(Q, t)
            REC.ident("D10.LS residual ~ 0", res, tol=1e-8)
            sv = torch.linalg.svdvals(Q)
            rk = int((sv > 1e-9 * sv.max()).sum()) if sv.numel() else 0
            err = _mx(alpha - 1.0 / len(H)) * len(H)
            if rk == len(H):
                nind += 1
                REC.ident("D10.LS coefficients = 1/|H_u| (linearly independent baskets)", err, tol=1e-7)
            else:
                ndep += 1
                nmis += int(err > 1e-6)
            REC.check("D10.LS coefficients = 1/|H_u| (all baskets)", err <= 1e-6)
    # constructed cases
    cases = []
    shx = dict(sizes=(2,), d=4, Ku=3, Kt=4)
    rng = np.random.default_rng([10, 77])

    def run(name, sizes, C, H, d, arm="noid"):
        D = Data.manual(sizes, C, [H], "bag")
        p, _ = make_params(rng, D, dict(shx, d=d), "shared", "unit")
        g = grid(p, D, arm)
        _, Eidi, _, Eidu = tabs(p, D.V)
        Q = g["qi"][H] - (Eidi[H] if arm == "ids" else 0.0)
        t = g["qu"][0] - (Eidu[0] if arm == "ids" else 0.0)
        alpha, res = ls_alpha(Q, t)
        cases.append(f"{name}: LS alpha = [{', '.join(f'{float(a):.4f}' for a in alpha)}] vs 1/|H| = {1 / len(H):.4f}; "
                     f"residual {res:.1e}; sum(alpha) = {float(alpha.sum()):.4f}")
        return alpha

    run("(i) bag, noid, items {g1},{g2},{g1,g2}", (2,), [[1, 0], [0, 1], [1, 1]], [0, 1, 2], 4)
    run("(i') same, ids arm (q_j - e_ID(j))", (2,), [[1, 0], [0, 1], [1, 1]], [0, 1, 2], 4, arm="ids")
    run("(iii) bag, noid, one word-less item {g1},{g2},{}", (2,), [[1, 0], [0, 1], [0, 0]], [0, 1, 2], 4)
    run("(iv) fixed 2x2 design (a1b1,a1b2,a2b1,a2b2): dependent but null space orthogonal to 1",
        (2, 2), [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]], [0, 1, 2, 3], 6)
    Dg = Data(rng, dict(sizes=(5, 6, 4), M=20, N=1, d=4), "fixed", basket=(7, 7))
    pg, _ = make_params(rng, Dg, dict(sizes=(5, 6, 4), d=4, Ku=3, Kt=4), "shared", "unit")
    gg = grid(pg, Dg, "noid")
    al, rs = ls_alpha(gg["qi"][Dg.baskets[0]], gg["qu"][0])
    cases.append(f"(ii) fixed, noid, |H_u| = 7 > d = 4: LS alpha = [{', '.join(f'{float(a):.4f}' for a in al)}] vs "
                 f"{1 / 7:.4f}; residual {rs:.1e}")
    sec.add("convex-combination / hull membership", V(REC.ok("D10.convex identity q_u(-e_ID) = mean_j q_j(-e_ID) (noid)",
                                                             "D10.convex identity q_u(-e_ID) = mean_j q_j(-e_ID) (ids)")),
            REC.s("D10.convex identity q_u(-e_ID) = mean_j q_j(-e_ID) (noid)", "D10.convex identity q_u(-e_ID) = mean_j q_j(-e_ID) (ids)"))
    sec.add("LS reconstruction 'gives exactly 1/|H_u| each, residual ~ 0'",
            "FALSIFIED" if nmis > 0 else "CONFIRMED",
            REC.s("D10.LS residual ~ 0", "D10.LS coefficients = 1/|H_u| (linearly independent baskets)",
                  "D10.LS coefficients = 1/|H_u| (all baskets)") +
            f"<br>random baskets: {nind} linearly independent (all exact), {ndep} rank-deficient, of which {nmis} give "
            f"LS coefficients != 1/|H_u|<br>constructed: " + "<br>".join(cases),
            "The residual is always ~0 (q_u is in the span), but the least-squares (minimum-norm) coefficients equal "
            "1/|H_u| only when the basket's compositions are linearly independent -- or, more precisely, when the null "
            "space of [q_j]_{j in H_u} is orthogonal to the all-ones vector. This needs |H_u| <= d and no word-sum "
            "dependencies; it fails for |H_u| > d (the realistic regime: baskets are routinely longer than d), for bag "
            "items whose word sums are linearly dependent ({g1},{g2},{g1,g2}: alpha = (2/9, 2/9, 4/9), which does not "
            "even sum to 1), and for word-less items in the noid arm (q_j = 0 gets alpha_j = 0). Correct statement: "
            "q_u = (1/|H_u|) sum_j q_j always (so q_u is in the convex hull), but LS on the basket recovers the "
            "1/|H_u| weights only under linear independence (or null-space-orthogonal-to-1); otherwise the convex "
            "certificate is non-unique and LS returns a different, possibly non-convex, solution.")


# ============================================================== D11
def test_D11():
    sec = Section("D11", "B_user is rank-inert but loss-active under BCE only")
    BS = (-1e3, -7.3, -1e-2, 0.5, 13.0, 1e3)
    for s, sh, lay, arm, scn, rng in iter_cfg(tag=11, seeds=SEEDS[:4]):
        for table in ("shared", "sep"):
            D = Data(rng, sh, lay)
            rows = Rows.sample(D, rng, 3)
            p, _ = make_params(rng, D, sh, table, scn)
            o = fwd_rows(p, D, arm, rows)
            S0 = o["S"].detach()
            sat = _mx(S0) > 700.0  # BCE terms underflow (softplus(-s) == 0 for s >~ 745)
            u0 = int(rows.users[0])
            sel = (rows.users == u0).to(torch.float64)
            for kind in LOSSES:
                L0 = loss_of(S0, kind)
                Sv0 = S0.clone().requires_grad_(True)
                (gS0,) = torch.autograd.grad(loss_of(Sv0, kind), Sv0)
                for b in BS:
                    for bvec in (b * sel, b * torch.tensor(rng.normal(size=D.N))[rows.users]):
                        L1 = loss_of(S0 + bvec[:, None], kind)
                        if kind == "bce":
                            # first-order expected change; below the test's resolution -> skip
                            pred = abs(float((gS0.sum(1) * bvec).sum()))
                            if sat or pred < 1e-8 * max(abs(float(L0)), 1e-300):
                                REC.skip("D11b.BCE loss changes with b")
                            else:
                                REC.differ("D11b.BCE loss changes with b", rel(L1, L0))
                        else:
                            REC.ident(f"D11a.{kind} loss unchanged under per-user shift b", rel(L1, L0))
                Sv = S0.clone().requires_grad_(True)
                (gS,) = torch.autograd.grad(loss_of(Sv, kind), Sv)
                bt = torch.zeros(D.N, requires_grad=True)
                (gb,) = torch.autograd.grad(loss_of(S0 + bt[rows.users][:, None], kind), bt)
                for u in torch.unique(rows.users).tolist():
                    rr = rows.users == u
                    den = float(gS[rr].abs().sum())
                    if kind == "softmax":
                        den = max(den, float(rr.sum()))  # softmax grad = p - e_0: operand scale is 1 per row
                    if kind == "bce" and sat:
                        REC.skip("D11b.BCE dL/db != 0")
                        continue
                    if den == 0:
                        REC.skip("D11b.BCE dL/db != 0" if kind == "bce" else f"D11a.{kind} dL/db = 0", "ident" if kind != "bce" else "differ")
                        continue
                    val = abs(float(gb[u])) / den
                    if kind == "bce":
                        if val > 1e-13:  # float noise floor ~1e-16
                            REC.differ("D11b.BCE dL/db != 0", val, thr=1e-13)
                        else:
                            # near-cancellation: re-evaluate dL/db = sum_rows [sigma(S+) - 1 + sum_n sigma(S_n)]
                            # in 60-digit decimal arithmetic on the float scores
                            getcontext().prec = 60
                            sg = lambda x: Decimal(1) / (Decimal(1) + (-Decimal(float(x))).exp())
                            ex = sum((sg(row[0]) - Decimal(1) + sum(sg(z) for z in row[1:])) for row in S0[rr])
                            dexact = float(abs(ex)) / den
                            REC.differ("D11b.BCE dL/db != 0", dexact, thr=0.0)
                            REC.differ("D11b.(info) near-cancelling BCE rows: exact |dL/db|/sum|dL/dS| (60-digit)", dexact, thr=0.0)
                    else:
                        REC.ident(f"D11a.{kind} dL/db = 0 (= B_user-routed gradient (dL/db) w_u)", val)
                # literal reading: directional derivative of L along w_u in q_u space (= along e_ID(u) in the ids arm)
                if kind != "bce":
                    wu = p["W_u"].sum(0)
                    qu0 = o["qu"].detach().clone().requires_grad_(True)
                    (gq,) = torch.autograd.grad(loss_of(fwd_rows(p, D, arm, rows, qu=qu0)["S"], kind), qu0)
                    for u in torch.unique(rows.users).tolist():
                        rr = rows.users == u
                        den = float(gq[rr].norm(dim=1).sum()) * float(wu.norm())
                        if den == 0:
                            REC.skip(f"D11a(literal).{kind} directional derivative along w_u != 0")
                            continue
                        REC.differ(f"D11a(literal).{kind} directional derivative along w_u != 0",
                                   abs(float((gq[rr] @ wu).sum())) / den)
            # (c) ranking
            g = grid(p, D, arm)
            dS = g["S"][:, :, None] - g["S"][:, None, :]
            for b in BS:
                Sb = g["S"] + b
                dB = Sb[:, :, None] - Sb[:, None, :]
                REC.check("D11c.no pairwise order reversal under +b (|b| <= 1e3)", not bool(((dS > 0) & (dB < 0)).any()))
                REC.check("D11c.(info) no ties created by rounding under +b (|b| <= 1e3)", not bool(((dS != 0) & (dB == 0)).any()))
    # float tie demonstration
    rng = np.random.default_rng([11, 5])
    D = Data(rng, SHAPES[0], "fixed")
    p, _ = make_params(rng, D, SHAPES[0], "shared", "unit")
    Sg = grid(p, D, "ids")["S"]
    bb = 1e17 * _mx(Sg)
    ties = sum(int(torch.unique(Sg[u] + bb).numel() < D.M) for u in range(D.N))
    ka = ["D11a.softmax loss unchanged under per-user shift b", "D11a.bpr loss unchanged under per-user shift b",
          "D11a.softmax dL/db = 0 (= B_user-routed gradient (dL/db) w_u)", "D11a.bpr dL/db = 0 (= B_user-routed gradient (dL/db) w_u)"]
    sec.add("(a) loss invariance and dL/db = 0", V(REC.ok(*ka)), REC.s(*ka),
            "b added to every candidate score of a training row, per user (one fixed user, and independent b_u for all "
            "users); |dL/db_u| normalised by sum |dL/dS| over that user's rows (softmax: by max(that, #rows), since "
            "dL/dS = p - e_0 has operand scale 1 per row and p_0 - 1 rounds to 0 when p_0 ~ 1). The gradient routed through the B_user "
            "component into q_u is (dL/db) * w_u, so it vanishes with dL/db.")
    kl = ["D11a(literal).softmax directional derivative along w_u != 0", "D11a(literal).bpr directional derivative along w_u != 0"]
    lit_nonzero = REC.ok(*kl)
    sec.add("(a) 'their gradients with respect to the direction w_u = W_u^T 1 in parameter space -- i.e. dL/db'",
            "FALSIFIED" if lit_nonzero else "CONFIRMED", REC.s(*kl),
            "The identification is wrong: moving parameters along w_u (e.g. e_ID(u) -> e_ID(u) + eps w_u, or q_u -> "
            "q_u + eps w_u in the noid arm) is NOT a pure B_user shift -- it also moves u^ by eps W_u w_u (so the "
            "item-dependent I-half changes through t*) and moves u* (cosine). The directional derivative of the "
            "softmax/BPR loss along w_u is therefore generically nonzero (normalised values above; tested for both "
            "arms/tables). No parameter direction realises a pure per-user constant shift in this model. Correct "
            "statement: dL/db = 0 for softmax/BPR, equivalently the component of the gradient that flows THROUGH the "
            "B_user term, (sum_i dL/dS_ui) w_u, is exactly 0 -- not the gradient along the direction w_u.")
    kb = ["D11b.BCE loss changes with b", "D11b.BCE dL/db != 0"]
    sec.add("(b)", V(REC.ok(*kb)), REC.s(*kb, "D11b.(info) near-cancelling BCE rows: exact |dL/db|/sum|dL/dS| (60-digit)"),
            "configs whose scores exceed |S| > 700 (the 1e3-scale sets) are skipped for BCE: there every BCE term "
            "under/overflows to an exactly linear or exactly zero softplus in float64, so a change of size e^-|S| is "
            "below float resolution. Mathematically dL/db = -sigma(-S+) + sum sigma(S-) is nonzero off a codim-1 set. "
            "dL/db is tested against the float noise floor (thr 1e-13 relative); cases below it are re-evaluated in 60-digit "
            "decimal arithmetic and counted as differing iff the exact value is nonzero; the finite-change test skips cases "
            "whose first-order change |b dL/db| is < 1e-8 |L|. Observation: in rows where a wrong-side positive and a "
            "wrong-side negative are both deep in the linear regime of softplus, their b-dependence cancels and dL/db is "
            "exponentially small (seen: ~1e-12 relative) -- nonzero, but B_user's BCE activity then rides only on the "
            "non-saturated terms.")
    sec.add("(c)", V(REC.ok("D11c.no pairwise order reversal under +b (|b| <= 1e3)")),
            REC.s("D11c.no pairwise order reversal under +b (|b| <= 1e3)", "D11c.(info) no ties created by rounding under +b (|b| <= 1e3)"),
            f"exact-arithmetic claim. Floating-point note: at b = 1e17 x max|S| the shifted scores collapse to ties "
            f"for {ties}/{D.N} users (rounding), so 'any b' holds mathematically, not in float64 for |b| >~ 1e16 x score spread.")


# ============================================================== report
def write_report(runtime, env):
    out = ["# dc09 SC.2 claims -- black-box toy check (report_sc2.md)", "",
           f"Generated by `check_claims_sc2.py` (implemented from `claims_spec_sc2.md` only). {env}. "
           f"Runtime {runtime:.1f} s, CPU.", "",
           "## Conventions and readings", "",
           "- Guarded cosine: cos(a,b) = a.b / (max(|a|,1e-8) max(|b|,1e-8)).",
           "- Training row (u, i+, negs): one q_u^LOO (basket H_u minus i+, mean over |H_u|-1, word part 0 if empty) feeds "
           "u* and u^ for every candidate of the row; negatives drawn from items outside H_u.",
           "- Gradient tests use each loss with and without the inclusion regularisers "
           "(0.3 R_proto_item - 0.2 R_batch_item + 0.25 R_proto_user + 0.15 R_batch_user over the row batch).",
           "- 'Exactly' / 'machine precision': relative error <= 1e-11 in float64, where the denominator is the magnitude "
           "of the operands that enter the computation (so pure floating-point cancellation is not mistaken for a failure; "
           "where a kappa factor is applied it is stated). 'Bit-identical' is tested with torch.equal in float64 and float32.",
           "- 'Differs generically': relative difference > 1e-9 on random instances (float64 rounding noise in these tests is <= ~1e-13, so 1e-9 separates a genuine difference from noise even at mixed scales where the true difference is ~1e-7 relative).",
           f"- Shapes (fields' vocab sizes, M, N, d, K_u, K_t): " + "; ".join(
               f"{s['name']}: {s['sizes']}, {s['M']}, {s['N']}, {s['d']}, {s['Ku']}, {s['Kt']}" for s in SHAPES) +
           ". Word ids are randomly permuted across fields. Bag layout: k_{i,phi} uniform in 0..min(6,|V_phi|) "
           "(distinct words); 'bagms' = bag layout with word multisets (k_{i,phi} in 0..6 drawn with replacement, so a word "
           "can repeat inside an item); baskets 1..6 items unless stated. Layout sets: fixed/bag/bagms wherever a claim is "
           "not restricted to the fixed layout.",
           "- Seeds 0-5 (subsets where noted); scale sets: unit, all 1e-3, all 1e3, mixA (E_w 1e3, ID 1e-3, P 1, W_u 1e-3, "
           "W_t 1e1), mixB (E_w 1e-3, ID 1e3, P 1e3, W_u 1e3, W_t 1e-3), all-negative entries, random per-block scale in "
           "1e-3..1e3.", "",
           "## Summary", "", "| claim | verdict |", "|---|---|"]
    for sec in SECTIONS:
        out.append(f"| {sec.cid} -- {sec.title} | {sec.overall()} |")
    out.append("")
    for sec in SECTIONS:
        out += [f"## {sec.cid} -- {sec.title}", "", f"**Overall: {sec.overall()}**", ""]
        for sub, verdict, detail, reading in sec.items:
            out.append(f"### {sec.cid}{sub if sub.startswith('(') else ' ' + sub} -- {verdict}")
            out.append("")
            out.append(f"- Measured: {detail}")
            if reading:
                out.append(f"- Reading / explanation: {reading}")
            out.append("")
        for n in sec.notes:
            out.append(f"- Note: {n}")
            out.append("")
    with open(os.path.join(HERE, "report_sc2.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    t0 = time.time()
    env = (f"python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}, "
           f"conda env {os.environ.get('CONDA_DEFAULT_ENV', '?')}")
    for fn in (test_D1, test_D2, test_D3, test_D4, test_D5, test_D6, test_D7, test_D8, test_D9, test_D10, test_D11):
        t = time.time()
        fn()
        print(f"{fn.__name__}: {time.time() - t:.1f}s -> {SECTIONS[-1].overall()}", flush=True)
    write_report(time.time() - t0, env)
    print(f"total {time.time() - t0:.1f}s; wrote {os.path.join(HERE, 'report_sc2.md')}")


if __name__ == "__main__":
    main()
