// Thompson Sampling with Normal-Inverse-Gamma prior
// Arms = prompts
// Each pull evaluates ONE random example
// Hard token budget
// Unknowns = mean and variance of prompt scores
// More we see a prompt, better we know its distribution (less variance)

import { z } from "npm:zod";
import { chatJSON, LLMConfig } from "./callOllama.ts";
import { LLM } from "@langchain/core/language_models/llms";
import { LLMChain } from "langchain/chains";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";

export type Prompt = { id: string; instruction: string };
export type BestPrompts = { best: Prompt; score: number; tokens: number };
export type EvalExample<E> = (
  instruction: string,
  ex: E
) => Promise<{ score: number; tokens: number }>;

export class TokenMeter {
  total = 0;
  add(n: number) {
    this.total += Math.max(0, n | 0);
  }
  snapshot() {
    return this.total;
  }
  can(budget?: number | null) {
    return budget == null || this.total < budget;
  }
}
function uuid() {
  return crypto.randomUUID();
}
function randint(n: number) {
  return Math.floor(Math.random() * n);
}

// Create a single mutation of a prompt
const MutSchema = z.object({ instruction: z.string().min(8).max(8000) });
export async function mutateOnce(
  config: LLMConfig,
  parent: string,
  meter: TokenMeter,
  guidance = "Rewrite to be clearer, shorter, and enforce returning ONLY valid JSON."
): Promise<{ instruction: string; tokens: number }> {
  const system =
    "You rewrite prompts. Preserve intent and output schema. No examples; be concise.";
  const user =
    `GUIDANCE: ${guidance}\n` +
    `Rewrite the instruction below. Keep the same JSON schema.\n---\n${parent}\n---\n` +
    `Return JSON: {"instruction":"<rewritten>"}`;
  const { data, tokens } = await chatJSON(
    config,
    [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    MutSchema
  );
  const out = data?.instruction?.trim() || parent;
  meter.add(tokens ?? 0);
  return { instruction: out, tokens: tokens ?? 0 };
}

// Normal Inverse Gamma = mean, variance prior/posterior
// Prior and posterior are conjugate this means we can simply update hyperparameters as we see data
// If they were'nt conjugate this would mean we would need to approximate the posterior to update it
type Posterior = {
    mu: number; // mean
    kappa: number; // inverse spread of mean
    alpha: number; // variance shape
    beta: number;  // variance scale
};

// Sample from normal and inverse gamma distributions
// mean is derived from variance's stddev
function normal(mean: number, variance: number) {
  const u = Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + Math.sqrt(Math.max(variance, 1e-12)) * z;
}

// Draw sample variance from inverse gamma
function invGamma(alpha: number, beta: number) {
  // d = shape, c = scale
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  let v: number;

  // while v is less than 0, re-sample from standard normal
  do {
    const u = Math.random();
    const w = Math.random();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    // v = (1 + c * z)^3
    v = Math.pow(1 + c * z, 3);
  } while (v <= 0);

  // return beta / (d * v) as sample from InvGamma
  const gammaSample = d * v;
  return beta / gammaSample;
}

export type TSOptions<E> = {
  config: LLMConfig;
  seeds: string[]; // at least one base instruction
  data: E[];
  evalExample: EvalExample<E>;
  budget: number | null; // hard token budget (mutations + eval)
  extraArms?: number; // how many mutated arms to add at start (3 for now)
  prior?: Posterior; // Normal inverse gamma prior hyperparameters
};

export async function tsOptimize<E>(opts: TSOptions<E>) {
  const { config, seeds, data, evalExample, budget, extraArms = 3 } = opts;
  const prior: Posterior = opts.prior ?? {
    mu: 0.5, // prior mean score
    kappa: 1e-3, // prior strength
    alpha: 1.0, // prior shape (flat prior of (1,1) to start)
    beta: 1.0, // prior scale
  };
  const meter = new TokenMeter();

  // Initialize candidate prompts
  const arms: Prompt[] = (
    seeds.length ? seeds : ['Return only {"label":"A|B"}']
  ).map((s) => ({ id: uuid(), instruction: s }));

  // Create extra mutated arms
  for (let i = 0; i < extraArms; i++) {
    const base = arms[0].instruction;
    const { instruction } = await mutateOnce(config, base, meter);
    if (!meter.can(budget)) break;
    arms.push({ id: uuid(), instruction });
  }

  // Add posterior for each arm
  const P = new Map<string, Posterior>();
  arms.forEach((a) => P.set(a.id, { ...prior }));

  // Track best by posterior mean
  let best = arms[0];
  let bestMu = P.get(best.id)!.mu;
  const bestPrompts: BestPrompts[] = [
    { best, score: bestMu, tokens: meter.snapshot() },
  ];

  // Loop until budget exhausted
  while (meter.can(budget)) {
    // Thompson sample each arm
    let chosen: Prompt | null = null;
    let bestTheta = -Infinity;

    for (const a of arms) {
      const p = P.get(a.id)!;
      const sigma2 = invGamma(p.alpha, p.beta);
      const theta = normal(p.mu, sigma2 / p.kappa);
      if (theta > bestTheta) {
        bestTheta = theta;
        chosen = a;
      }
    }

    if (!chosen) break;

    // Evaluate at random a given example
    const idx = randint(data.length);
    const { score, tokens } = await evalExample(chosen.instruction, data[idx]);
    meter.add(tokens);

    // Posterior update after observing score
    // More observations == more certain about mean and variance
    const post = P.get(chosen.id)!;
    const k1 = post.kappa + 1;
    const mu1 = (post.kappa * post.mu + score) / k1;
    const a1 = post.alpha + 0.5;
    const b1 = post.beta + (post.kappa * (score - post.mu) ** 2) / (2 * k1);
    P.set(chosen.id, { mu: mu1, kappa: k1, alpha: a1, beta: b1 });

    // Track best by posterior mean
    // Update best if improved
    best = arms.reduce(
      (acc, cur) => (P.get(cur.id)!.mu > P.get(acc.id)!.mu ? cur : acc),
      best
    );
    bestMu = P.get(best.id)!.mu;
    bestPrompts.push({ best, score: bestMu, tokens: meter.snapshot() });

    if (!meter.can(budget)) break;
  }

  return { best, bestPrompts, meter, posterior: P };
}
