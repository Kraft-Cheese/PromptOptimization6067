import { z } from "npm:zod";
import { chatJSON } from "./callOllama.ts";
import type { MCExample, BoolQExample, GSM8KExample } from "./datasets.ts";
import type { EvalExample } from "./ape.ts";

// Multiple choice evals for PiQA and HellaSwag
const AB = z.object({ label: z.enum(["0", "1"]) });
const ABCD = z.object({ label: z.enum(["0", "1", "2", "3"]) });

// For 2 or 4 choice multiple choice tasks
export function makeMCEvaluator(
  model: string,
  numChoices: 2 | 4 = 2
): EvalExample<MCExample> {
  const schema = numChoices === 2 ? AB : ABCD;
  
  return async (instruction: string, ex: MCExample) => {
    const choiceLabels = numChoices === 2 ? ['0', '1'] : ['0', '1', '2', '3'];
    const optionsText = ex.choices
      .slice(0, numChoices)
      .map((choice, i) => `${choiceLabels[i]}) ${choice}`)
      .join('\n');
    
    const user = 
      `Question: ${ex.question}\n\n` +
      `Options:\n${optionsText}\n\n` +
      `Return JSON: {"label": "${choiceLabels.join('|')}"}`;
    
    const { data, tokens } = await chatJSON(
      model,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ],
      schema
    );
    
    const predicted = data?.label ?? null;
    const score = predicted === ex.correct ? 1.0 : 0.0;
    
    return { score, tokens };
  };
}

// Evals for PIQA and HellaSwag (2 and 4 choices)
export function makePIQAEvaluator(model: string): EvalExample<MCExample> {
  return makeMCEvaluator(model, 2);
}

export function makeHellaSwagEvaluator(model: string): EvalExample<MCExample> {
  return makeMCEvaluator(model, 4);
}

// Boolean evaluator
const YesNo = z.object({ answer: z.enum(["yes", "no"]) });

// Issue yes/no answers for BoolQ
export function makeBoolQEvaluator(model: string): EvalExample<BoolQExample> {
  return async (instruction: string, ex: BoolQExample) => {
    const user = 
      `Passage: ${ex.passage}\n\n` +
      `Question: ${ex.question}\n\n` +
      `Return JSON: {"answer": "yes"} or {"answer": "no"}`;
    
    const { data, tokens } = await chatJSON(
      model,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ],
      YesNo
    );
    
    const predicted = data?.answer;
    const goldAnswer = ex.answer ? "yes" : "no";
    const score = predicted === goldAnswer ? 1.0 : 0.0;
    
    return { score, tokens };
  };
}

// For GSM8K as it is numeric answer
const NumericAnswer = z.object({ 
  answer: z.number(),
  reasoning: z.string().optional()
});

// Numeric answer evaluator for GSM8K
export function makeGSM8KEvaluator(model: string): EvalExample<GSM8KExample> {
  return async (instruction: string, ex: GSM8KExample) => {
    const user = 
      `Problem: ${ex.question}\n\n` +
      `Return JSON: {"answer": <number>, "reasoning": "<optional>"}`;
    
    const { data, tokens } = await chatJSON(
      model,
      [
        { role: "system", content: instruction },
        { role: "user", content: user }
      ],
      NumericAnswer
    );
    
    const predicted = data?.answer ?? null;
    
    // Floating point tolerance
    const tolerance = 0.01;
    const score = 
      predicted !== null && Math.abs(predicted - ex.answer) < tolerance 
        ? 1.0 
        : 0.0;
    
    return { score, tokens };
  };
}