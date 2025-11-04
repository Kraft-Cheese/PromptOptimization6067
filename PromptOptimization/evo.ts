// Binary Tournament Evolution

import { z } from "npm:zod";
import { chatText, LLMConfig } from "./callOllama.ts";

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

const mutatorStats = new Map<string, { uses: number; improvements: number }>();
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
    const { response, tokens } = await chatText(
      config,
      [{ role: "system", content: system }, { role: "user", content: user }]
    );
    const instruction = response.trim() || parent;
    meter.add(tokens ?? 0);
    return { instruction, tokens: tokens ?? 0 };
  };
}

// A few defaults inspired by Ape + PromptBreeder paper
export const DEFAULT_MUTATORS: Mutator[] = [
  makeMutator("format_strict", "Stress: output ONLY valid JSON; no explanations."),
  makeMutator("tie_break", "If multiple labels seem plausible, choose the MOST SPECIFIC according to the class set."),
  makeMutator("cautious", "Prefer safety and common-sense plausibility when unsure."),
  makeMutator("reason_silent", "Think step-by-step INTERNALLY and NEVER reveal your reasoning."),
  makeMutator("concise", "Make the instruction shorter and more direct while preserving all constraints."),
  makeMutator("specific", "Add specific details and examples to clarify what's expected."),
  makeMutator("step_by_step", "Add explicit step-by-step guidance for solving the task."),
  makeMutator("error_prevention", "Add warnings about common mistakes and how to avoid them."),
  makeMutator("confidence", "Add guidance about expressing certainty levels in ambiguous cases."),
  makeMutator("edge_cases", "Add handling for edge cases and unusual inputs."),
  makeMutator("clarity", "Rephrase for maximum clarity and reduce ambiguity."),
  makeMutator("authoritative", "Rewrite with a more authoritative and expert tone."),
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

  const mutatorStats = new Map<string, { uses: number; improvements: number }>();
  for (let i = 0; i < mutators.length; i++) {
    mutatorStats.set(`mutator_${i}`, { uses: 0, improvements: 0 });
  }

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

    const mutatorIdx = Math.floor(Math.random() * mutators.length);
    const mut = mutators[mutatorIdx];
    const mutKey = `mutator_${mutatorIdx}`;
    const stats = mutatorStats.get(mutKey)!;
    stats.uses++;

    // replace the loser with a mutated child of the winner
    // const mut = pick(mutators);
    const { instruction: childInst } = await mut(config, winner.instruction, meter);
    if (!meter.can(budget)) break;

    // Create child prompt with new uuid and replace in population
    const child: Prompt = { id: uuid(), instruction: childInst };
    pop[loserIdx] = child;

    // Evaluate child fitness and update best if improved
    const sChild = await fitness(child);
    if (!meter.can(budget)) break;

    if (sChild > bestScore) { 
        best = child; 
        bestScore = sChild; 
        stats.improvements++;
        if (globalThis.Deno?.env?.get("DEBUG")) {
        console.log(`\nImprovement from ${mutKey}: ${bestScore.toFixed(3)}`);
        }
    }
    bestPrompts.push({ best, score: bestScore, tokens: meter.snapshot() });
  }

  if (globalThis.Deno?.env?.get("DEBUG")) {
    mutatorStats.forEach((stats, key) => {
      const successRate = stats.uses > 0 ? (stats.improvements / stats.uses * 100).toFixed(1) : "0.0";
      console.log(`${key}: ${stats.uses} uses, ${stats.improvements} improvements (${successRate}%)`);
    });
  }

  return { best, bestPrompts, meter };
}
