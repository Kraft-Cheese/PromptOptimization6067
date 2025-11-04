import { z } from "npm:zod";
import { LLMConfig } from "./callOllama.ts";
import type { MCExample, BoolQExample, GSM8KExample } from "./datasets.ts";
import type { EvalExample } from "./ape.ts";

import { chatText} from "./callOllama.ts";

// For 2 or 4 choice multiple choice tasks
export function makeMCEvaluator(
  config: LLMConfig,
  numChoices: 2 | 4 = 2
): EvalExample<MCExample> {
  return async (instruction: string, ex: MCExample) => {
    const choiceLabels = numChoices === 2 ? ['A', 'B'] : ['A', 'B', 'C', 'D'];
    const optionsText = ex.choices
      .slice(0, numChoices)
      .map((choice, i) => `${choiceLabels[i]}) ${choice}`)
      .join('\n');

    const user =
      `Question: ${ex.question}\n\n` +
      `Options:\n${optionsText}\n\n` +
      `Respond with ONLY the letter of the correct answer (${choiceLabels.join(' or ')}). No explanation.`;

    const { response, tokens } = await chatText(
      config,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ]
    );

    // Extract first uppercase letter from response
    const match = response.match(/[A-D]/);
    const predicted = match ? match[0] : null;
    
    const score = predicted === ex.correct ? 1.0 : 0.0;

    if (globalThis.Deno?.env?.get("DEBUG")) {
      console.log("\n--- DEBUG ---");
      console.log("Question:", ex.question.slice(0, 60));
      console.log("Correct:", ex.correct);
      console.log("Response:", response.trim());
      console.log("Predicted:", predicted);
      console.log("Match:", score === 1.0);
    }

    return { score, tokens };
  };
}

// Evals for PIQA and HellaSwag (2 and 4 choices)
export function makePIQAEvaluator(config: LLMConfig): EvalExample<MCExample> {
  return makeMCEvaluator(config, 2);
}

export function makeHellaSwagEvaluator(config: LLMConfig): EvalExample<MCExample> {
  return makeMCEvaluator(config, 4);
}

// Boolean evaluator for BoolQ
export function makeBoolQEvaluator(config: LLMConfig): EvalExample<BoolQExample> {
  return async (instruction: string, ex: BoolQExample) => {
    const user =
      `Passage: ${ex.passage}\n\n` +
      `Question: ${ex.question}\n\n` +
      `Respond with ONLY "yes" or "no". No explanation.`;

    const { response, tokens } = await chatText(
      config,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ]
    );

    const cleaned = response.trim().toLowerCase();
    const predicted = cleaned.includes('yes') ? 'yes' : 
                     cleaned.includes('no') ? 'no' : null;
    
    const goldAnswer = ex.answer ? "yes" : "no";
    const score = predicted === goldAnswer ? 1.0 : 0.0;

    if (globalThis.Deno?.env?.get("DEBUG")) {
      console.log("\n--- DEBUG ---");
      console.log("Question:", ex.question.slice(0, 60));
      console.log("Correct:", goldAnswer);
      console.log("Response:", response.trim());
      console.log("Predicted:", predicted);
      console.log("Match:", score === 1.0);
    }

    return { score, tokens };
  };
}

// Numeric answer evaluator for GSM8K
export function makeGSM8KEvaluator(config: LLMConfig): EvalExample<GSM8KExample> {
  return async (instruction: string, ex: GSM8KExample) => {
    const user =
      `${ex.question}\n\n` +
      `Respond with ONLY the final numerical answer. No explanation.`;

    const { response, tokens } = await chatText(
      config,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ]
    );

    // Extract last number from response
    const numbers = response.match(/-?\d+\.?\d*/g);
    const predicted = numbers ? parseFloat(numbers[numbers.length - 1]) : null;
    
    // Floating point tolerance
    const tolerance = 0.01;
    const score =
      predicted !== null && Math.abs(predicted - ex.answer) < tolerance
        ? 1.0
        : 0.0;

    if (globalThis.Deno?.env?.get("DEBUG")) {
      console.log("\n--- DEBUG ---");
      console.log("Question:", ex.question.slice(0, 60));
      console.log("Correct:", ex.answer);
      console.log("Response:", response.trim());
      console.log("Predicted:", predicted);
      console.log("Match:", score === 1.0);
    }

    return { score, tokens };
  };
}

export async function saveResults(
  filename: string,
  results: unknown,
  prettify = true
): Promise<void> {
  try {
    const json = JSON.stringify(results, null, prettify ? 2 : 0);
    await Deno.writeTextFile(filename, json);
    console.log(`Saved to ${filename}`);
  } catch (error) {
    console.error(`Failed to save results to ${filename}:`, error);
  }
}

export async function loadResults<T>(filename: string): Promise<T | null> {
  try {
    const json = await Deno.readTextFile(filename);
    return JSON.parse(json) as T;
  } catch (error) {
    console.error(`Failed to load results from ${filename}:`, error);
    return null;
  }
}

export interface AlgorithmResult {
  name: string;
  bestScore: number;
  tokensUsed: number;
  iterations: number;
  efficiency: number; // score per 1000 tokens
}

export function compareResults(
  ape: { bestPrompts: Array<{ score: number; tokens: number }> },
  evo: { bestPrompts: Array<{ score: number; tokens: number }> },
  ts: { bestPrompts: Array<{ score: number; tokens: number }> }
): AlgorithmResult[] {
  const results: AlgorithmResult[] = [];

  const apeFinal = ape.bestPrompts[ape.bestPrompts.length - 1];
  results.push({
    name: "APE",
    bestScore: apeFinal.score,
    tokensUsed: apeFinal.tokens,
    iterations: ape.bestPrompts.length,
    efficiency: (apeFinal.score / apeFinal.tokens) * 1000,
  });

  const evoFinal = evo.bestPrompts[evo.bestPrompts.length - 1];
  results.push({
    name: "Evolution",
    bestScore: evoFinal.score,
    tokensUsed: evoFinal.tokens,
    iterations: evo.bestPrompts.length,
    efficiency: (evoFinal.score / evoFinal.tokens) * 1000,
  });

  const tsFinal = ts.bestPrompts[ts.bestPrompts.length - 1];
  results.push({
    name: "Thompson",
    bestScore: tsFinal.score,
    tokensUsed: tsFinal.tokens,
    iterations: ts.bestPrompts.length,
    efficiency: (tsFinal.score / tsFinal.tokens) * 1000,
  });

  return results;
}

export async function safeOptimize<T>(
  name: string,
  optimizeFn: () => Promise<T>
): Promise<T | null> {
  try {
    console.log(`\nStarting ${name}`);
    const result = await optimizeFn();
    console.log(`${name} completed successfully`);
    return result;
  } catch (error) {
    console.error(`${name} failed:`, error);
    return null;
  }
}