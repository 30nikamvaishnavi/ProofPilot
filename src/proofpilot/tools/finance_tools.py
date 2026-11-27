"""Quantitative-finance tools: closed form verified by Monte Carlo / independent implementation."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import norm

from ..schemas import VerifyResult
from .base import Tool, ToolOutput

MC_PATHS = 400_000


def bs_price(s: float, k: float, t: float, r: float, sigma: float, kind: str) -> tuple[float, float]:
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    disc = math.exp(-r * t)
    if kind == "call":
        return s * norm.cdf(d1) - k * disc * norm.cdf(d2), float(norm.cdf(d1))
    return k * disc * norm.cdf(-d2) - s * norm.cdf(-d1), float(norm.cdf(d1) - 1.0)


class BSArgs(BaseModel):
    S: float = Field(gt=0)
    K: float = Field(gt=0)
    T: float = Field(gt=0, le=50)
    r: float = Field(ge=-0.5, le=1.0)
    sigma: float = Field(gt=0, le=5)
    option_type: Literal["call", "put"] = "call"


class BlackScholes(Tool):
    name = "black_scholes"
    domain = "finance"
    description = "European option price and delta under Black-Scholes."
    arg_hint = 'S, K: spot and strike (>0); T: years; r: risk-free rate; sigma: volatility; option_type: "call"|"put"'
    assumptions = [
        "Black-Scholes: lognormal prices, constant volatility and rate, no dividends, European exercise.",
    ]
    ArgsModel = BSArgs

    def run(self, args: BSArgs) -> ToolOutput:
        price, delta = bs_price(args.S, args.K, args.T, args.r, args.sigma, args.option_type)
        value = {"price": round(price, 6), "delta": round(delta, 6)}
        return ToolOutput(value, f"{args.option_type} price = {value['price']:.4f}, delta = {value['delta']:.4f}")

    def verify(self, args: BSArgs, output: ToolOutput) -> VerifyResult:
        rng = np.random.default_rng(12345)
        z = rng.standard_normal(MC_PATHS // 2)
        z = np.concatenate([z, -z])  # antithetic variates
        st = args.S * np.exp((args.r - 0.5 * args.sigma**2) * args.T + args.sigma * math.sqrt(args.T) * z)
        payoff = np.maximum(st - args.K, 0.0) if args.option_type == "call" else np.maximum(args.K - st, 0.0)
        disc = math.exp(-args.r * args.T) * payoff
        mc = float(disc.mean())
        se = float(disc.std(ddof=1) / math.sqrt(len(disc)))
        claimed = float(output.value["price"])
        c, _ = bs_price(args.S, args.K, args.T, args.r, args.sigma, "call")
        p, _ = bs_price(args.S, args.K, args.T, args.r, args.sigma, "put")
        parity = abs((c - p) - (args.S - args.K * math.exp(-args.r * args.T)))
        ok = bool(abs(mc - claimed) <= 4 * se + 1e-6 and parity < 1e-8)
        return VerifyResult(
            ok=ok,
            method=f"Monte Carlo ({MC_PATHS} antithetic paths, 4 s.e.) + put-call parity",
            detail=f"closed form={claimed:.6f}, MC={mc:.6f} ± {se:.6f}, parity error={parity:.2g}",
        )


class PortfolioArgs(BaseModel):
    weights: list[float] = Field(min_length=1, max_length=50)
    cov: list[list[float]]


class PortfolioVariance(Tool):
    name = "portfolio_variance"
    domain = "finance"
    description = "Portfolio variance and volatility w' Σ w for given weights and covariance matrix."
    arg_hint = "weights: list of floats; cov: square symmetric positive semi-definite matrix"
    assumptions = ["Covariance matrix is symmetric positive semi-definite; weights are used as given (not normalised)."]
    ArgsModel = PortfolioArgs

    @staticmethod
    def _check(args: PortfolioArgs) -> tuple[np.ndarray, np.ndarray]:
        w = np.array(args.weights, dtype=float)
        c = np.array(args.cov, dtype=float)
        n = len(w)
        if c.shape != (n, n):
            raise ValueError("cov must be a square matrix matching the number of weights")
        if not np.allclose(c, c.T, atol=1e-10):
            raise ValueError("cov must be symmetric")
        if float(np.linalg.eigvalsh(c).min()) < -1e-9:
            raise ValueError("cov must be positive semi-definite")
        return w, c

    def run(self, args: PortfolioArgs) -> ToolOutput:
        w, c = self._check(args)
        var = float(w @ c @ w)
        vol = math.sqrt(max(var, 0.0))
        return ToolOutput({"variance": var, "volatility": vol}, f"variance = {var:.8g}, volatility = {vol:.8g}")

    def verify(self, args: PortfolioArgs, output: ToolOutput) -> VerifyResult:
        self._check(args)
        n = len(args.weights)
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += args.weights[i] * args.weights[j] * args.cov[i][j]
        claimed = float(output.value["variance"])
        ok = math.isclose(total, claimed, rel_tol=1e-9, abs_tol=1e-12) and math.isclose(
            math.sqrt(max(total, 0.0)), float(output.value["volatility"]), rel_tol=1e-9, abs_tol=1e-12
        )
        return VerifyResult(ok=ok, method="independent double-sum implementation", detail=f"claimed={claimed:.10g}, double-sum={total:.10g}")
