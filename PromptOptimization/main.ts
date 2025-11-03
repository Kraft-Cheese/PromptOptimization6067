import { apeOptimize } from './ape.ts';
import { evoOptimize } from './evo.ts';
import { tsOptimize } from './thompson.ts';
import {
  loadPIQA,
  loadHellaSwag,
  loadBoolQ,
  loadGSM8K
} from './datasets.ts';
import {
  makePIQAEvaluator,
  makeHellaSwagEvaluator,
  makeBoolQEvaluator,
  makeGSM8KEvaluator
} from './evals.ts';
import { LLMConfig } from './callOllama.ts';

const MODEL = "qwen3:8b";
const BUDGET = 5000;

const ollamaConfig: LLMConfig = {
  provider: "openai",
  model: "gpt-3.5-turbo",
  apiKey: process.env.OPENAI_API_KEY
};

const openaiConfig: LLMConfig = {
  provider: "openai",
  model: "gpt-4",
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://custom-endpoint.com/v1"
};

const llmConfig: LLMConfig = {
  provider: "ollama",
  model: "qwen3:8b",
};

// Base instructions for each task
const INSTRUCTIONS = {
  piqa:
    'You are a classifier for physical commonsense reasoning. ' +
    'Analyze both options carefully and choose the more practical solution. ' +
    'Return ONLY valid JSON: {"label": "A"} or {"label": "B"}.',

  hellaswag:
    'You are a commonsense reasoning classifier. ' +
    'Read the context and determine which ending makes most sense. ' +
    'Return ONLY valid JSON: {"label": "A|B|C|D"}.',

  boolq:
    'You are a reading comprehension system. ' +
    'Read the passage and answer the question truthfully. ' +
    'Return ONLY valid JSON: {"answer": "yes"} or {"answer": "no"}.',

  gsm8k:
    'You are a math problem solver. ' +
    'Solve the problem step by step. ' +
    'Return ONLY valid JSON: {"answer": <number>, "reasoning": "<optional>"}.'
};

async function runDataset(
  name: string,
  loadData: () => Promise<any[]>,
  makeEvaluator: (config: LLMConfig) => any,
  baseInstruction: string,
  numExamples = 30
) {
  console.log(`DATASET: ${name.toUpperCase()}`);

  const data = await loadData();
  const dataSlice = data.slice(0, numExamples);
  const evalExample = makeEvaluator(llmConfig);

  console.log(`Loaded ${dataSlice.length} examples`);
  console.log(`Base instruction: ${baseInstruction.slice(0, 80)}...`);

  // APE
  console.log("Running APE");
  const apeResult = await apeOptimize({
    config: llmConfig,
    baseInstruction,
    N: 5,
    data: dataSlice,
    evalExample,
  });
  console.log(`Score: ${apeResult.bestPrompts[0].score.toFixed(3)}, Tokens: ${apeResult.bestPrompts[0].tokens}`);

  // Evolution
  console.log("Running Evolution");
  const evoResult = await evoOptimize({
    config: llmConfig,
    seeds: [baseInstruction],
    data: dataSlice,
    evalExample,
    budget: BUDGET,
  });
  const evoFinal = evoResult.bestPrompts[evoResult.bestPrompts.length - 1];
  console.log(`Score: ${evoFinal.score.toFixed(3)}, Tokens: ${evoFinal.tokens}, Rounds: ${evoResult.bestPrompts.length}`);

  // Thompson
  console.log("Running Thompson Sampling");
  const tsResult = await tsOptimize({
    config: llmConfig,
    seeds: [baseInstruction],
    data: dataSlice,
    evalExample,
    budget: BUDGET,
    extraArms: 3,
  });
  const tsFinal = tsResult.bestPrompts[tsResult.bestPrompts.length - 1];
  console.log(`Score: ${tsFinal.score.toFixed(3)}, Tokens: ${tsFinal.tokens}, Pulls: ${tsResult.bestPrompts.length}`);

  // Save results
  const results = {
    dataset: name,
    numExamples: dataSlice.length,
    baseInstruction,
    ape: { bestPrompts: apeResult.bestPrompts, best: apeResult.best },
    evo: { bestPrompts: evoResult.bestPrompts, best: evoResult.best },
    ts: { bestPrompts: tsResult.bestPrompts, best: tsResult.best }
  };

  await Deno.writeTextFile(
    `results_${name}.json`,
    JSON.stringify(results, null, 2)
  );

  console.log(`Saved to results_${name}.json`);
}

async function main() {
  console.log("PROMPT OPTIMIZATION");
  console.log(`Model: ${MODEL}`);
  console.log(`Budget: ${BUDGET} tokens per algorithm`);

  try {
    // Run all datasets
    await runDataset("piqa", loadPIQA, makePIQAEvaluator, INSTRUCTIONS.piqa, 30);
    await runDataset("hellaswag", loadHellaSwag, makeHellaSwagEvaluator, INSTRUCTIONS.hellaswag, 30);
    await runDataset("boolq", loadBoolQ, makeBoolQEvaluator, INSTRUCTIONS.boolq, 30);
    await runDataset("gsm8k", loadGSM8K, makeGSM8KEvaluator, INSTRUCTIONS.gsm8k, 20); // Math is slower

    console.log("ALL EXPERIMENTS COMPLETED");
    console.log("\nResults saved:");
    console.log("results_piqa.json");
    console.log("results_hellaswag.json");
    console.log("results_boolq.json");
    console.log("results_gsm8k.json");

  } catch (error) {
    console.error("Error:", error);
  }
}

if (import.meta.main) {
  main();
}