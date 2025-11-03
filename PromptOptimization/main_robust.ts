import { apeOptimize } from './ape.ts';
import { evoOptimize } from './evo.ts';
import { tsOptimize } from './thompson.ts';
import {
  loadPIQA,
  loadHellaSwag,
  loadBoolQ,
  loadGSM8K,
  validateDataset
} from './datasets.ts';
import {
  makePIQAEvaluator,
  makeHellaSwagEvaluator,
  makeBoolQEvaluator,
  makeGSM8KEvaluator,
  saveResults,
  compareResults,
  safeOptimize,
  loadResults
} from './evals.ts';
import {
  DummyEvaluator
} from './dummy.ts';
import {
  LLMConfig,
} from './callOllama.ts';
import {
  ProgressTracker,
} from './track.ts';

const CONFIG = {
  // LLM Provider "ollama" or "openai"
  provider: (Deno.env.get("LLM_PROVIDER") || "ollama") as "ollama" | "openai",
  model: Deno.env.get("MODEL") || "qwen3:8b" | "mistral:7b",
  apiKey: Deno.env.get("OPENAI_API_KEY"), // Only needed for OpenAI

  // Optimization parameters
  budget: parseInt(Deno.env.get("BUDGET") || "5000"),
  numExamples: parseInt(Deno.env.get("NUM_EXAMPLES") || "30"),
  apeParaphrases: 5,
  evoExtraArms: 3,

  // Testing mode with mock evals
  useDummyEval: Deno.env.get("DUMMY_MODE") === "true",

  // Result caching
  useCache: Deno.env.get("USE_CACHE") === "true",
  cacheDir: "./cache"
};

const llmConfig: LLMConfig = {
  provider: CONFIG.provider,
  model: CONFIG.model,
  apiKey: CONFIG.apiKey
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

// Create cache
async function ensureCacheDir() {
  if (CONFIG.useCache) {
    try {
      await Deno.mkdir(CONFIG.cacheDir, { recursive: true });
    } catch (error) {
      if (!(error instanceof Deno.errors.AlreadyExists)) {
        throw error;
      }
    }
  }
}

function getCachePath(dataset: string): string {
  return `${CONFIG.cacheDir}/results_${dataset}.json`;
}

async function runDataset(
  name: string,
  loadData: () => Promise<any[]>,
  makeEvaluator: (config: LLMConfig) => any,
  baseInstruction: string,
  numExamples: number = CONFIG.numExamples
) {
  console.log(`DATASET: ${name.toUpperCase()}`);

  // Check cache first
  if (CONFIG.useCache) {
    const cached = await loadResults(getCachePath(name));
    if (cached) {
      console.log(`Loaded cached results for ${name}`);
      return cached;
    }
  }

  // Load and validate dataset (requirement: min 8 examples)
  const data = await loadData();
  validateDataset(data, 8, name);
  const dataSlice = data.slice(0, numExamples);

  console.log(`Loaded ${dataSlice.length} examples`);
  console.log(`Base instruction: ${baseInstruction.slice(0, 80)}...`);
  console.log(`Provider: ${CONFIG.provider}, Model: ${CONFIG.model}`);
  console.log(`Budget: ${CONFIG.budget} tokens per algorithm\n`);

  // Create evaluator (or dummy for testing)
  let evalExample;
  if (CONFIG.useDummyEval) {
    console.log("Using DUMMY evaluator for testing");
    const dummy = new DummyEvaluator(6); // 6 possible arms
    evalExample = (inst: string, ex: any) => dummy.evaluate(inst, ex);
  } else {
    evalExample = makeEvaluator(llmConfig);
  }

  // ============================================================================
  // ALGORITHM 1: APE (Automatic Prompt Engineering)
  // ============================================================================
  console.log("Running APE");

  const apeResult = await safeOptimize("APE", async () => {
    const tracker = new ProgressTracker(CONFIG.apeParaphrases + 1);

    const result = await apeOptimize({
      config: CONFIG.model,
      baseInstruction,
      N: CONFIG.apeParaphrases,
      data: dataSlice,
      evalExample,
    });

    tracker.update(CONFIG.apeParaphrases + 1);
    return result;
  });

  if (apeResult) {
    const apeFinal = apeResult.bestPrompts[0];
    console.log(`Score: ${apeFinal.score.toFixed(3)}, Tokens: ${apeFinal.tokens}`);
    console.log(`Best: ${apeResult.best.instruction.slice(0, 100)}...\n`);
  }

  console.log("Running Evolution");

  const evoResult = await safeOptimize("Evolution", async () => {
    return await evoOptimize({
      model: CONFIG.model,
      seeds: [baseInstruction],
      data: dataSlice,
      evalExample,
      budget: CONFIG.budget,
    });
  });

  if (evoResult) {
    const evoFinal = evoResult.bestPrompts[evoResult.bestPrompts.length - 1];
    console.log(
      `Score: ${evoFinal.score.toFixed(3)}, ` +
      `Tokens: ${evoFinal.tokens}, ` +
      `Rounds: ${evoResult.bestPrompts.length}`
    );
    console.log(`  Best: ${evoResult.best.instruction.slice(0, 100)}...\n`);
  }


  console.log("Running Thompson Sampling");

  const tsResult = await safeOptimize("Thompson Sampling", async () => {
    return await tsOptimize({
      model: CONFIG.model,
      seeds: [baseInstruction],
      data: dataSlice,
      evalExample,
      budget: CONFIG.budget,
      extraArms: CONFIG.evoExtraArms,
    });
  });

  if (tsResult) {
    const tsFinal = tsResult.bestPrompts[tsResult.bestPrompts.length - 1];
    console.log(
      `Score: ${tsFinal.score.toFixed(3)}, ` +
      `Tokens: ${tsFinal.tokens}, ` +
      `Pulls: ${tsResult.bestPrompts.length}`
    );
    console.log(`Best: ${tsResult.best.instruction.slice(0, 100)}...\n`);
  }

  console.log("COMPARISON");

  if (apeResult && evoResult && tsResult) {
    const comparison = compareResults(apeResult, evoResult, tsResult);

    console.log("\nAlgorithm     | Final Score | Tokens  | Iterations | Efficiency");
    console.log("-".repeat(70));

    for (const result of comparison) {
      console.log(
        `${result.name.padEnd(14)} | ` +
        `${result.bestScore.toFixed(4).padEnd(12)} | ` +
        `${result.tokensUsed.toString().padEnd(8)} | ` +
        `${result.iterations.toString().padEnd(10)} | ` +
        `${result.efficiency.toFixed(2)} pts/1k tok`
      );
    }

    console.log();
  }

  // Save results
  const results = {
    dataset: name,
    numExamples: dataSlice.length,
    baseInstruction,
    config: {
      provider: CONFIG.provider,
      model: CONFIG.model,
      budget: CONFIG.budget,
      useDummyEval: CONFIG.useDummyEval
    },
    ape: apeResult ? { bestPrompts: apeResult.bestPrompts, best: apeResult.best } : null,
    evo: evoResult ? { bestPrompts: evoResult.bestPrompts, best: evoResult.best } : null,
    ts: tsResult ? { bestPrompts: tsResult.bestPrompts, best: tsResult.best } : null
  };

  const filename = `results_${name}.json`;
  await saveResults(filename, results);

  if (CONFIG.useCache) {
    await saveResults(getCachePath(name), results);
  }

  return results;
}

async function main() {
  console.log("PROMPT OPTIMIZATION FRAMEWORK");
  console.log(`Provider: ${CONFIG.provider.toUpperCase()}`);
  console.log(`Model: ${CONFIG.model}`);
  console.log(`Budget: ${CONFIG.budget} tokens per algorithm`);
  console.log(`Examples per dataset: ${CONFIG.numExamples}`);
  if (CONFIG.useDummyEval) {
    console.log("RUNNING IN DUMMY MODE (no real LLM calls)");
  }
  console.log();

  await ensureCacheDir();

  try {
    // Run all datasets
    const datasets = [
      { name: 'piqa', loader: loadPIQA, eval: makePIQAEvaluator, instruction: INSTRUCTIONS.piqa },
      { name: 'hellaswag', loader: loadHellaSwag, eval: makeHellaSwagEvaluator, instruction: INSTRUCTIONS.hellaswag },
      { name: 'boolq', loader: loadBoolQ, eval: makeBoolQEvaluator, instruction: INSTRUCTIONS.boolq },
      { name: 'gsm8k', loader: loadGSM8K, eval: makeGSM8KEvaluator, instruction: INSTRUCTIONS.gsm8k },
    ];

    const allResults = [];

    for (const dataset of datasets) {
      const result = await runDataset(
        dataset.name,
        dataset.loader,
        dataset.eval,
        dataset.instruction,
        dataset.name === 'gsm8k' ? 20 : CONFIG.numExamples // Math is slower
      );
      allResults.push(result);
    }

    // Save combined results
    await saveResults('results_all.json', {
      config: CONFIG,
      timestamp: new Date().toISOString(),
      results: allResults
    });

    console.log("ALL EXPERIMENTS COMPLETED");
    console.log("\nResults saved:");
    for (const dataset of datasets) {
      console.log(`results_${dataset.name}.json`);
    }
    console.log(`results_all.json`);

  } catch (error) {
    console.error("\nError:", error);
    console.error("\nStack trace:", error.stack);
    Deno.exit(1);
  }
}

if (import.meta.main) {
  // Parse command line arguments
  const args = Deno.args;

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Prompt Optimization Framework

Usage:
  deno run --allow-all main_robust.ts [options]

Options:
  --help, -h           Shows help
  --provider <name>    LLM provider: "ollama" or "openai" (default is ollama)
  --model <name>       Model name (default is qwen3:8b)
  --budget <n>         Token budget per algorithm (default is 5000)
  --examples <n>       Number of examples per dataset (default is 30)
  --dummy              Mock evals (no LLM calls)
  --cache              Enable result caching
  --dataset <name>     Run only specific dataset of the 4 (piqa, hellaswag, boolq, gsm8k)

Environment Variables:
  LLM_PROVIDER         Same as --provider
  MODEL                Same as --model
  OPENAI_API_KEY       Required for OpenAI provider
  BUDGET               Same as --budget
  NUM_EXAMPLES         Same as --examples
  DUMMY_MODE           Set to "true" for dummy mode
  USE_CACHE            Set to "true" to enable caching

Examples:
  # Run with Ollama (local)
  deno run --allow-all main_robust.ts

  # Run with OpenAI
  deno run --allow-all main_robust.ts --provider openai --model gpt-3.5-turbo

  # Quick test with dummy evaluators
  deno run --allow-all main_robust.ts --dummy --examples 10

  # Run specific dataset only
  deno run --allow-all main_robust.ts --dataset piqa --examples 50
    `);
    Deno.exit(0);
  }

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--provider":
        CONFIG.provider = args[++i] as "ollama" | "openai";
        break;
      case "--model":
        CONFIG.model = args[++i];
        break;
      case "--budget":
        CONFIG.budget = parseInt(args[++i]);
        break;
      case "--examples":
        CONFIG.numExamples = parseInt(args[++i]);
        break;
      case "--dummy":
        CONFIG.useDummyEval = true;
        break;
      case "--cache":
        CONFIG.useCache = true;
        break;
    }
  }

  main();
}