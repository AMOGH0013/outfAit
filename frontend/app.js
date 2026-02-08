async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);

  const contentType = res.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await res.json().catch(() => null);
  } else {
    payload = await res.text().catch(() => null);
  }

  if (!res.ok) {
    const detail =
      payload && typeof payload === "object" && payload.detail
        ? payload.detail
        : payload;
    throw new Error(`${res.status} ${res.statusText}: ${detail || "Request failed"}`);
  }

  return payload;
}

function renderImage(url) {
  if (!url) return "";
  return url;
}

function fmtDateISO(d) {
  const dt = new Date(d);
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const dd = String(dt.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function summarizeExplanation(expl) {
  if (!expl || typeof expl !== "object") return "Explanation unavailable.";

  const positives = [];
  const cautions = [];

  const fitScore = Number(expl.fit_score ?? 0);
  const embScore = Number(expl.embedding_diversity_score ?? 0);
  const fbBias = Number(expl.feedback_bias ?? 0);
  const styleScore = Number(expl.style_affinity_score ?? 0);

  const reasoning = expl.reasoning && typeof expl.reasoning === "object" ? expl.reasoning : {};

  // Core rule-based reasons (if present)
  if (reasoning.color) positives.push("the colors pair well");
  if (reasoning.novelty) positives.push("it avoids items you wore recently");

  // Fit preference (bounded)
  if (fitScore > 0.01) positives.push("it matches your fit preference");
  if (fitScore < -0.01) cautions.push("it may not match your fit preference");

  // Embedding diversity
  if (embScore > 0.01) positives.push("it adds visual variety vs recently worn outfits");
  if (embScore < -0.01) cautions.push("it looks similar to outfits you wore recently");

  // Feedback bias
  if (fbBias > 0.01) positives.push("your past feedback boosts similar combinations");
  if (fbBias < -0.01) cautions.push("your past feedback penalizes similar combinations");

  // Style affinity
  if (styleScore > 0.01) positives.push("it matches your learned style preferences");
  if (styleScore < -0.01) cautions.push("it may not match your learned style preferences");

  if (positives.length === 0 && cautions.length === 0) {
    return "This is a balanced, neutral recommendation.";
  }

  const first = positives.length
    ? `This outfit works because ${positives.join(", ")}.`
    : "This outfit is recommended, but with some tradeoffs.";

  const second = cautions.length ? `Note: ${cautions.join(", ")}.` : "";
  return [first, second].filter(Boolean).join(" ");
}

function computePersonalizationStatus(profile, hasFeedback) {
  const p = profile && typeof profile === "object" ? profile : {};

  const features = {
    fit_preference: Boolean(p.fit_preference),
    body_shape: Boolean(p.body_shape),
    skin_tone: Boolean(p.skin_tone),
    feedback: Boolean(hasFeedback),
  };

  const score =
    (features.fit_preference ? 1 : 0) +
    (features.body_shape ? 1 : 0) +
    (features.skin_tone ? 1 : 0) +
    (features.feedback ? 1 : 0);

  let level = "Basic";
  if (score === 2) level = "Medium";
  if (score >= 3) level = "High";

  return { level, score, features };
}
