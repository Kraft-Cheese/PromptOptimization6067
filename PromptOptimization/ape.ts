// LLM-based paraphrasing of the instruction

import { z } from "npm:zod";
import { chatJSON } from "./callOllama.ts";

// prompt is id + the prompt's text
export type Prompt = { id: string; instruction: string };

// Track best current prompt, its score, and tokens used
export type BestPrompt = { best: Prompt; score: number; tokens: number };

// Evaluator: given an instruction and an example, return scalar score in [0,1] and tokens used.
export type EvalExample<E> = (
  instruction: string,
  ex: E
) => Promise<{ score: number; tokens: number }>;

// Track tokens used for budget 
export class TokenMeter {
  total = 0;
  add(n: number) { this.total += Math.max(0, n|0); }
  snapshot() { return this.total; }
}

// UUID for unique prompt IDs to track
function uuid() { return crypto.randomUUID(); }

// Schema of a paraphrased prompt
const ParaphraseSchema = z.object({
  instruction: z.string().min(8).max(8000),
});

/**
 * Paraphrase the given prompt
 * Counts tokens used by the paraphrasing call
 */
export async function paraphraseInstruction(
  model: string,
  base: string,
  meter: TokenMeter,
  style = "Be concise. Keep the same schema. Emphasize: return ONLY valid JSON."
): Promise<{ instruction: string; tokens: number }> {
  const system =
    "You rewrite prompts. Preserve intent and constraints; reduce verbosity; DO NOT change the output schema.";
  const user =
    `Rewrite this instruction with the same meaning and constraints. Do not add examples.\n` +
    `STYLE: ${style}\n\n---\n${base}\n---\n` +
    `Return JSON: {"instruction": "<rewritten>"} (no extra fields).`;

  const { data, tokens } = await chatJSON(
    model,
    [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    ParaphraseSchema
  );

  const rewritten = data?.instruction?.trim() || base;
  meter.add(tokens ?? 0);
  return { instruction: rewritten, tokens: tokens ?? 0 };
}

export type APEOptions<E> = {
  model: string;
  baseInstruction: string; // starting instruction to paraphrase
  N: number; // number of paraphrases to try
  data: E[]; // data is the eval examples for: PIQA, HellaSwag, BoolQ, GSM8K      
  evalExample: EvalExample<E>; // evaluator function for the dataset
};

export async function apeOptimize<E>(opts: APEOptions<E>) {
  const { model, baseInstruction, N, data, evalExample } = opts;
  const meter = new TokenMeter();

  // Create candidates: base + N paraphrases and set uuids + track total tokens (max per prompt is 128)
  const candidates: Prompt[] = [{ id: uuid(), instruction: baseInstruction }];
  for (let i = 0; i < N; i++) {
    const { instruction } = await paraphraseInstruction(model, baseInstruction, meter);
    candidates.push({ id: uuid(), instruction });
  }

  // Evaluate each candidate on the dataset + examples
  let best = candidates[0];
  let bestScore = -1;

  // Loop through candidates and evaluate
  for (const p of candidates) {
    let sum = 0;

    // Loop through eval examples
    for (let i = 0; i < data.length; i++) {
      const { score, tokens } = await evalExample(p.instruction, data[i]);
      meter.add(tokens); // update total tokens
      sum += score;
    }
    const avg = sum / Math.max(1, data.length); // average score over examples
    if (avg > bestScore) { 
      best = p; bestScore = avg; // update best prompt if improved
    }
  }

  // Return best prompt, its score, and token usage
  const bestPrompts: BestPrompt[] = [{ best, score: bestScore, tokens: meter.snapshot() }];
  return { best, bestPrompts, meter };
}
