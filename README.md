# Simple Prompt Optimisation
Three key techniques that have emerged in the literature: 
- LLM-based paraphrasing of prompts (APE), 
evolutionary algorithms like PromptBreeder,
and multi-armed bandits optimization (this time with Thompson sampling).
1.) Each of these approaches is implemented in a simplified form, in Typescript.
2.) They are called on a set of seed prompt(s),
    keep track of token usage,
  plotted on an “accuracy” or performance curve across generations (y-axis is performance of best prompt, x-axis is number of tokens used).

Benchmarks used : PiQA, HellaSwag, BoolQ, GMS8K

Compatible Providers : OpenAI, Ollama (local)

Implementations


1. A naïve approach that just paraphrases the base instruction, akin to APE.
  •  One mutator that you to mutate the base instruction N
times. You run all N prompts through the evaluation, each against *all* test
examples and take the average score. The “winning” prompt is the one with
the highest average.
  • Keep track of the tokens used, but no fixed max in
advance.
2. An evolutionary algorithm that uses basic binary tournament genetic
algorithm, akin to PromptBreeder, but not as complex:
  • Fixed set of “mutators” no hyper-mutation
  • Initial prompt(s) that you can mutate to seed the population
  • Binary tournament algorithm: each round, pick two prompts at
random from the population and measure their “fitness” (enter each of them
into the LLM and score the outputs across *all* input examples). Picks the
“winner” prompt and overrides the “loser” prompt with a mutated copy of the
winner. Continue to do this until a pre-specified budget (max number of tokens
used) is reached.
  • Keeps track of the best performance and token usage per “round” and store
these figures in a list (a list of tuples, (best prompt, score, token usage)).
Returned in addition to the best prompt.
3. An approach that uses Thompson sampling with a numeric evaluator, for a
single set of prompts and one fixed model. See this tutorial, which may be
helpful context.
  • To seed the initial population of prompts (“arms”), uses mutators from #2
(evolutionary algorithm) to generate an initial set.
  • Score the prompt on only *one* randomly selected
example each “pull” of the arm. That is, Thomson sampling is used to find
the “arm” (prompt) to “pull” (try), and pulling the arm is selecting a random
example input and testing it against that. This will drastically lower token
usage at expense of probabilistic sampling.
  • Use the Normal-inverse-gamma distribution for your prior. We use NIG here
because we don’t know the variance nor the mean of the distribution of
output quality (we don’t know if it’s normally distributed either, but we assume
this because we need to simplify somewhere).
  • Keep track of the best performance and token usage per “round” and store
these figures in a list (a list of tuples, (best prompt so far, best estimated score
so far, token usage)). Return this list in addition to the best prompt, which will
make it easier to plot what happened.
The latter two optimization functions must take in a “budget” as max token
usage, which is a good proxy for cost. These optimization algorithms must exit
immediately after the token budget has been reached. It is fine to estimate token usage
using tiktoken. Note: Thompson sampling is not a fixed budget algorithm and could
potentially continue forever, therefore we must exit manually and be careful to do so. If
no budget is specified, default to something “reasonable.”

Jupyter Notebooks (Typescript Kernel, Python Kernel)
1. Setup and Problem: Describe the problem you chose to optimize a prompt over,
and define all the initial test prompt(s), input examples, and evaluation function(s).
(Remember that you should only have a single “overall” evaluation function that
produces a numeric score, which could call other evaluators but sums them.)
2. Optimization algorithms: Define, in turn, each algorithm, separating them with
helpful ##headers.
3. Running, plotting, and comparing results: Run each algorithm and plot the
results for each algorithm alongside the “best prompt” with its score. You might
want to store the results somewhere so you can retrieve them without re-running.
- In this section, produce an overall “comparison plot” between all three

methods, with a legend for each method. The y-axis is best performance, and x-
axis is token usage per “round” (the APE algorithm will just be a single point on

this graph).
4. Reflection: Briefly, describe in words: what can you conclude about the trade-offs
between methods? Which method would you choose to move forward with?
Finally, how would you improve the algorithm(s) or develop your own algorithm to
improve performance
