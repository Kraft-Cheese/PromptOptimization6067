// Binary Tournament Evolution

import { z } from "npm:zod";
import { chatJSON, LLMConfig } from "./callOllama.ts";

export type Prompt = { id: string; instruction: string };
export type BestPrompt = { best: Prompt; score: number; tokens: number };
export type EvalExample<E> = (
  instruction: string,
  ex: E
) => Promise<{ score: number; tokens: number }>;

export class TokenMeter {
  total = 0;
  add(n: number) { this.total += Math.max(0, n|0); }
  snapshot() { return this.total; }
  can(budget?: number | null) { return budget == null || this.total < budget; }
}
function uuid() { return crypto.randomUUID(); }
function pick<T>(xs: T[]) { return xs[Math.floor(Math.random() * xs.length)]; }

// ---- Mutators (LLM rewrites of the instruction) ----
const MutSchema = z.object({ instruction: z.string().min(8).max(8000) });

export type Mutator = (
  config: LLMConfig,
  parentInstruction: string,
  meter: TokenMeter
) => Promise<{ instruction: string; tokens: number }>;

// Create a mutator using meta prompting
function makeMutator(title: string, guidance: string): Mutator {
  return async (config, parent, meter) => {
    const system =
      "You rewrite prompts. Preserve intent and output schema. Apply the guidance. No examples; be concise.";
    const user =
      `GUIDANCE: ${guidance}\n` +
      `Rewrite the instruction below. Keep the same JSON schema and constraints.\n---\n${parent}\n---\n` +
      `Return JSON: {"instruction":"<rewritten>"}`;
    const { data, tokens } = await chatJSON(
      config,
      [{ role: "system", content: system }, { role: "user", content: user }],
      MutSchema
    );
    const out = data?.instruction?.trim() || parent;
    meter.add(tokens ?? 0);
    return { instruction: out, tokens: tokens ?? 0 };
  };
}

// A few defaults inspired by Ape + PromptBreeder paper
export const DEFAULT_MUTATORS: Mutator[] = [
  makeMutator("format_strict", "Stress: output ONLY valid JSON; no explanations."),
  makeMutator("tie_break", "If multiple labels seem plausible, choose the MOST SPECIFIC according to the class set."),
  makeMutator("cautious", "Prefer safety and common-sense plausibility when unsure."),
  makeMutator("reason_silent", "Think step-by-step INTERNALLY and NEVER reveal your reasoning."),
];

type FitnessCacheKey = string;

export type EvoOptions<E> = {
  config: LLMConfig;
  seeds: string[]; // Original prompts to evolve from
  data: E[]; // Eval examples for: PIQA, HellaSwag, BoolQ, GSM8K
  evalExample: EvalExample<E>;
  budget: number | null; // hard token budget (counts mutators + eval)
  mutators?: Mutator[]; // default provided
};

export async function evoOptimize<E>(opts: EvoOptions<E>) {
  const { config, seeds, data, evalExample, budget, mutators = DEFAULT_MUTATORS } = opts;
  const meter = new TokenMeter();

  // Initialize set of prompts
  const pop: Prompt[] = (seeds.length ? seeds : ['Return only {"label":"A|B"}'])
    .map(s => ({ id: uuid(), instruction: s }));

  // Cache fitness results
  const fit = new Map<FitnessCacheKey, number>();
  const keyOf = (p: Prompt) => p.instruction;

  async function fitness(p: Prompt): Promise<number> {
    const k = keyOf(p);
    if (fit.has(k)) return fit.get(k)!;

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const { score, tokens } = await evalExample(p.instruction, data[i]);
      meter.add(tokens);
      sum += score;

      if (!meter.can(budget)) break; // stop early if budget blown during eval
    }
    const avg = sum / Math.max(1, data.length);
    fit.set(k, avg);
    return avg;
  }

  // From original prompts find the best as starting point
  let best = pop[0];
  let bestScore = await fitness(best);
  const bestPrompts: BestPrompt[] = [{ best, score: bestScore, tokens: meter.snapshot() }];

  // Tournament loop until budget reached
  while (meter.can(budget)) {
    const a = pick(pop), b = pick(pop);
    const fa = await fitness(a);
    if (!meter.can(budget)) break;
    const fb = await fitness(b);
    if (!meter.can(budget)) break;

    const winner = fa >= fb ? a : b;
    const loserIdx = pop.findIndex(x => x.id === (fa >= fb ? b.id : a.id));

    // replace the loser with a mutated child of the winner
    const mut = pick(mutators);
    const { instruction: childInst } = await mut(config, winner.instruction, meter);
    if (!meter.can(budget)) break;

    // Create child prompt with new uuid and replace in population
    const child: Prompt = { id: uuid(), instruction: childInst };
    pop[loserIdx] = child;

    // Evaluate child fitness and update best if improved
    const sChild = await fitness(child);
    if (!meter.can(budget)) break;

    if (sChild > bestScore) { best = child; bestScore = sChild; }
    bestPrompts.push({ best, score: bestScore, tokens: meter.snapshot() });
  }

  return { best, bestPrompts, meter };
}
