type JsonObject = Record<string, any>;
const UNITS = 1_000_000_000_000n;
type Fraction = { n: bigint; d: bigint };
function gcd(a: bigint, b: bigint): bigint {
  a = a < 0n ? -a : a;
  while (b) {
    const r = a % b;
    a = b;
    b = r;
  }
  return a;
}
function fraction(n: bigint, d: bigint): Fraction {
  const g = gcd(n, d);
  return { n: n / g, d: d / g };
}
function decimal(input: unknown): Fraction {
  if (typeof input !== "string" && typeof input !== "number")
    throw new Error("Expected decimal input");
  let s = String(input);
  if (s.length > 1024) throw new Error("Oversized decimal input");
  let sign = 1n;
  if (s.startsWith("-")) {
    sign = -1n;
    s = s.slice(1);
  } else if (s.startsWith("+")) s = s.slice(1);
  const exponentAt = Math.max(s.indexOf("e"), s.indexOf("E"));
  let exponent = 0;
  if (exponentAt >= 0) {
    const exponentText = s.slice(exponentAt + 1);
    const digits =
      exponentText.startsWith("+") || exponentText.startsWith("-")
        ? exponentText.slice(1)
        : exponentText;
    if (!digits || [...digits].some((c) => c < "0" || c > "9"))
      throw new Error("Invalid decimal exponent");
    exponent = Number(exponentText);
    s = s.slice(0, exponentAt);
    if (!Number.isInteger(exponent) || Math.abs(exponent) > 1024)
      throw new Error("Invalid decimal exponent");
  }
  const parts = s.split(".");
  if (
    parts.length > 2 ||
    !parts.join("") ||
    [...parts.join("")].some((c) => c < "0" || c > "9")
  )
    throw new Error("Invalid decimal input");
  const scale = (parts[1]?.length ?? 0) - exponent,
    n = BigInt(parts.join("")) * sign;
  return fraction(
    scale < 0 ? n * 10n ** BigInt(-scale) : n,
    scale > 0 ? 10n ** BigInt(scale) : 1n,
  );
}
function add(a: Fraction, b: Fraction): Fraction {
  return fraction(a.n * b.d + b.n * a.d, a.d * b.d);
}
function multiply(a: Fraction, b: Fraction): Fraction {
  return fraction(a.n * b.n, a.d * b.d);
}
function equal(a: Fraction, b: Fraction): boolean {
  return a.n * b.d === b.n * a.d;
}
function positive(a: Fraction): Fraction {
  if (a.n <= 0n) throw new Error("Non-positive capitalization input");
  return a;
}
export function capWeights(
  manifest: JsonObject,
  scores: JsonObject[],
  cutoff: number,
  weighting: "market_cap" | "market_cap_rank" = "market_cap",
  missingCapitalization: "exclude" | "block" = "exclude",
): JsonObject {
  if (!["market_cap", "market_cap_rank"].includes(weighting)) throw new Error("Unsupported index weighting");
  if (!["exclude", "block"].includes(missingCapitalization)) throw new Error("Unsupported missing-capitalization policy");
  const byId = new Map(scores.map((s) => [s.security_id, s]));
  if (
    byId.size !== scores.length ||
    scores.length !== manifest.securities.length ||
    manifest.securities.some((r: JsonObject) => !byId.has(r.security_id))
  )
    throw new Error("Expected exactly one score per frozen security");
  const selected = manifest.securities.filter(
    (r: JsonObject) => byId.get(r.security_id)!.score >= cutoff,
  );
  if (!selected.length)
    return {
      status: "needs_data",
      issues: [{ error: "no_members_meet_requested_cutoff" }],
      weights: [],
      requested_count: 0,
      included_count: 0,
      excluded_count: 0,
      excluded: [],
    };
  const issues: JsonObject[] = [],
    unavailable: JsonObject[] = [],
    caps = new Map<string, Fraction>(),
    issuers = new Set();
  for (const row of selected) {
    const cap = row.capitalizations?.issuer_total;
    if (cap?.status !== "ready") {
      unavailable.push({
        security_id: row.security_id,
        ticker: row.ticker,
        error: "capitalization_unavailable",
        data_issues: row.issues ?? [],
      });
      continue;
    }
    const shares = positive(decimal(row.fundamentals.latest_shares.value)),
      price = positive(decimal(row.price.underlying_price_usd)),
      conversion = row.listing_conversion ?? {};
    if (
      !equal(shares, decimal(cap.shares)) ||
      !equal(price, decimal(cap.price_per_share_usd)) ||
      !equal(
        decimal(conversion.factor ?? "1"),
        decimal(cap.listing_share_factor ?? "1"),
      )
    )
      throw new Error("Capitalization input binding mismatch");
    let equivalent: Fraction;
    if (conversion.class_components) {
      let count = decimal("0"),
        total = decimal("0");
      for (const component of conversion.class_components) {
        const units = decimal(component.shares);
        count = add(count, units);
        total = add(
          total,
          multiply(
            units,
            positive(decimal(component.listing_equivalents_per_share)),
          ),
        );
      }
      if (!equal(count, shares))
        throw new Error("Class conversion does not cover share count");
      equivalent = positive(total);
    } else
      equivalent = multiply(
        shares,
        positive(decimal(conversion.factor ?? "1")),
      );
    if (!equal(equivalent, decimal(cap.listing_equivalent_shares)))
      throw new Error("Listing-equivalent share binding mismatch");
    const value = multiply(equivalent, price);
    if (!equal(value, decimal(cap.market_cap_usd)))
      throw new Error("Capitalization derivation mismatch");
    caps.set(row.security_id, value);
    const cik = row.identity?.cik;
    if (issuers.has(cik))
      issues.push({
        error: "issuer_total_would_be_counted_multiple_times",
        cik,
      });
    issuers.add(cik);
  }
  const excluded = missingCapitalization === "exclude" ? unavailable : [];
  if (missingCapitalization === "block") issues.push(...unavailable);
  if (issues.length || !caps.size)
    return {
      status: "needs_data",
      issues: issues.length ? issues : unavailable,
      weights: [],
      requested_count: selected.length,
      included_count: 0,
      excluded_count: excluded.length,
      excluded,
    };
  const capOrder = [...caps.keys()].sort((a, b) => {
    const x = caps.get(a)!, y = caps.get(b)!;
    const difference = x.n * y.d - y.n * x.d;
    return difference > 0n ? -1 : difference < 0n ? 1 : a < b ? -1 : a > b ? 1 : 0;
  });
  const ranks = new Map(capOrder.map((id, i) => [id, i + 1]));
  if (weighting === "market_cap_rank") {
    for (const [id, rank] of ranks) caps.set(id, fraction(BigInt(caps.size - rank + 1), 1n));
  }
  let total = decimal("0");
  for (const cap of caps.values()) total = add(total, cap);
  const units = new Map<string, bigint>(),
    remainders = new Map<string, Fraction>();
  let allocated = 0n;
  for (const [id, cap] of caps) {
    const exact = fraction(cap.n * total.d * UNITS, cap.d * total.n),
      floor = exact.n / exact.d;
    units.set(id, floor);
    allocated += floor;
    remainders.set(id, fraction(exact.n % exact.d, exact.d));
  }
  const ordered = [...caps.keys()].sort((a, b) => {
    const left = remainders.get(a)!,
      right = remainders.get(b)!,
      delta = left.n * right.d - right.n * left.d;
    return delta > 0n ? -1 : delta < 0n ? 1 : a < b ? -1 : a > b ? 1 : 0;
  });
  for (let n = 0; n < Number(UNITS - allocated); n++) {
    const id = ordered[n]!;
    units.set(id, units.get(id)! + 1n);
  }
  return {
    status: "ready",
    weighting,
    ...(weighting === "market_cap_rank" ? {
      rank_formula: "(N - market_cap_rank + 1) / (N * (N + 1) / 2)",
      rank_ties: "security_id_ascending",
    } : {}),
    capitalization_basis: "issuer_total",
    integer_weight_units: Number(UNITS),
    rounding: "largest_remainder_then_security_id",
    requested_count: selected.length,
    included_count: caps.size,
    excluded_count: excluded.length,
    excluded,
    weights: [...units.keys()].sort().map((id) => {
      const count = units.get(id)!,
        r = manifest.securities.find((r: JsonObject) => r.security_id === id);
      return {
        security_id: id,
        ticker: r.ticker,
        company_name: r.fundamentals?.company_name ?? r.ticker,
        ...(weighting === "market_cap_rank" ? {
          market_cap_rank: ranks.get(id),
          rank_units: caps.size - ranks.get(id)! + 1,
          market_cap_usd: r.capitalizations.issuer_total.market_cap_usd,
        } : {}),
        score: byId.get(id)!.score,
        ...(byId.get(id)!.confidence !== undefined ? {confidence:byId.get(id)!.confidence} : {}),
        weight_units: Number(count),
        weight: `${count / UNITS}.${(count % UNITS).toString().padStart(12, "0")}`,
      };
    }),
  };
}
