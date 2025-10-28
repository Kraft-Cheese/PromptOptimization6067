# Prompt Optimization 6067
The goal of this assignment is not to “solve” prompt optimization, but to get familiar
with three key techniques that have emerged in the literature: 
- LLM-based paraphrasing of prompts (APE), 
evolutionary algorithms like PromptBreeder,
and multi-armed bandits optimization (this time with Thompson sampling).
1.) You will implement each of these approaches in a simplified form, in Typescript.
2.) You will then call them on a set of seed prompt(s),
    keep track of token usage,
  plot an “accuracy” or performance curve across generations (y-axis is performance of best prompt, x-axis is number of tokens used).

This assignment has a more lax structure than the previous one. 
You are not given a specific target to optimize for, and need to choose your own. 
You might look up benchmarks online and take a small subset of the benchmark (for cost reasons) to
target (for instance, GSM8K word problems is a common one, yet the latest models perform very well on this so you might deliberately choose worse/smaller models to
test on). 
Alternatively, you can invent your own situation. 
One interesting one is from ChainForge examples—the task to “only delete words from this paragraph without affecting grammaticality”—which LLMs always struggle with. 
For your chosen use case, you must develop evaluator that makes sense: 
  a function which takes the output of the model and produces a numeric score. 
For this task, if you have multiple evaluation criteria, you should take a “rubric” approach that adds up individual scores and produces an overall “grade” to optimize over.

Background
Prompt optimization is a hot topic these days, meant to automate the tedious and
error-prone process of prompt engineering. 
The many frequent and onerous proclamations of prompt engineering’s “death” have been resoundingly misplaced, however: manual prompt engineering, for better or worse, remains as important as ever. 
Recent results in scientific literature show that manual prompting efforts continue
to exceed auto-optimizations from popular frameworks [x,y]. 
When you look closely enough on hyped solutions that claim to “solve” prompting, you will see they have not actually replaced manual prompting, just obscured it further (e.g., just disingenuously reworded “prompt” to something like “signature”).

You should already have some intuition for why some manual prompt engineering
remains critical. Shankar et al.’s work on EvalGen identified the criteria drift phenomenon, which stipulates that it is impossible for humans to fully specify their evaluation criteria without first seeing some model outputs. 
That is, prompt engineering for virtually all important tasks requires iteration and manual review. 
It is not something that can ever be fully automated, only accelerated with smart human-in- the-loop mechanisms and better models.
Thus, maximal acceleration of prompt engineering should be the aim at any serious (black-boxed) prompt optimization effort. 
As a corollary to criteria drift, such an effort must be accompanied by an interface which allow users to specify the “prompt searchspace”: the “holes” or input variables for each prompt, that specify the finite parameters to search over. 
(I mean “interface” here broadly: it could be a domain- specific language (DSL), programming language, command line interface, or GUI.)

In my opinion, prompt optimization has yet to be addressed well in a user-centered way. 
This is shocking, considering how much money is flooding the space, but I hypothesize that it arrives from the general disdain among engineers for embracing the
messy iteration that is commonplace to the fields of HCI and design. 
A second plausible contributor is a pressure to capitalize on AI hype, through a tendency to claim “prompt engineering is dead” and hide/obscure the actual prompting that is inevitably going on. 
The longer you exist in this space, the more you will learn to see past the immediate hype of pre-prints and interrogate whether and to what extent these
approaches actually bear fruit.
Many prompt optimization algorithms now exist. 
However, they are often siloed, with no public implementation or a detached open-source repository that is hard to use and unify with other approaches. 
Early algorithms also universally ignore the problems of cost, latency, and especially model choice, which in actual scenarios are always used by decision-makers.
A proper solution to prompt optimization would not view it strictly as an optimization problem with a universal “right” answer. There are instead multiple
algorithms, and what we want are interfaces and approaches to help the user:
  a) specify the “search space” in an effective manner, and
  b) choose a proper algorithm for that search complexity, given their input data and their budget (upper bound on cost)
For instance, if the search space is only four concrete prompts in total over one model,
then a brute-force approach, where all prompts are tried, is best. 
However, the search space quickly explodes especially when chaining prompt templates over large datasets and many models. 
Hence, in this case, one would will need to choose an algorithm that uses techniques from optimization, such as random sampling.
This assignment makes a number of simplifications reminiscent of the literature. 
For instance, you will practically speaking have an overall “template” that you feed inputs into, and only be optimizing a base “instruction.” 
You also will only optimize over one model. 
However, keep in mind that these are simplifying the situation deliberately, and that in practice, prompt optimization should take into account multiple models.

Instructions

In this unique assignment, you will implement prompt optimization in three “algorithms”, each of increasing complexity:
1. A naïve approach that just paraphrases the base instruction, akin to APE.
  •  You have just one mutator that you use to mutate the base instruction N
times. You run all N prompts through the evaluation, each against *all* test
examples and take the average score. The “winning” prompt is the one with
the highest average.
  • For this algorithm, keep track of the tokens used, but don’t fix a max in
advance.
2. An evolutionary algorithm that uses basic binary tournament genetic
algorithm, akin to PromptBreeder, but not as complex:
  • You have a set of “mutators” fixed in advance—different ways of asking an
LLM to rewrite an instruction. (Ignore the “hyper-mutation” idea of
PromptBreeder here.)
  • You have initial prompt(s) that you can mutate to seed the population
  • You set up a binary tournament algorithm: each round, pick two prompts at
random from the population and measure their “fitness” (enter each of them
into the LLM and score the outputs across *all* input examples). Pick the
“winner” prompt and override the “loser” prompt with a mutated copy of the
winner. Continue to do this until a pre-specified budget (max number of tokens
used) is reached.
  • Keep track of the best performance and token usage per “round” and store
these figures in a list (a list of tuples, (best prompt, score, token usage)).
Return this list in addition to the best prompt, which will make it easier to plot
what happened.
3. An approach that uses Thompson sampling with a numeric evaluator, for a
single set of prompts and one fixed model. See this tutorial, which may be
helpful context.
  • To seed the initial population of prompts (“arms”), use your mutators from #2
(evolutionary algorithm) to generate an initial set.
  • In this approach, instead of running each prompt through *all* the input
examples, you will score the prompt on only *one* randomly selected
example each “pull” of the arm. That is, you use Thomson sampling to find
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
Input examples: Cost will rise the more examples you have. I ask you to have at least
8 different “representative” inputs (rows of your dataset) to try your prompt optimization
algorithm over. (Typically, we’d separate things out into a train and test set, but we’re
simplifying here, so you’re only plotting the “training” data.)
Model choice: I would highly recommend choosing a weaker and cheaper model to
run tests on, and simply stick to that model for this assignment.
Format: You are asked to implement your solution in Python inside a single Jupyter
notebook. Each optimization algorithm should have a single entry point—a function—
that passes in the necessary parameters as arguments.
Recommendation: Use dummy calls when testing!
Because the cost of prompt optimization is high, while building and testing your
solution, you are encouraged to use dummy calls—i.e., you aren’t actually calling an
LLM but using random number generators to output a dummy “evaluation result” with
fixed probability. You would assign these probabilities to each “arm” in advance, and
have your “evaluation function” just look at the arm index and draw with the ground
truth probability for that arm. Your solution needs to work for a real LLM, but this allows
you to do most of the assignment without making an API call, if you prefer.
Note: Cite anything you amend
If you adopt any open-source code, you are required to mention this and link to the
code alongside its license (if it is not licensed, then you aren’t allowed to use it).

Deliverable
You will submit a single Jupyter Notebook file with any plotted outputs already
produced and visible in the notebook without re-running the code. The latter is very

important: without the plots visible, I will not be able to give you a fair grade. The
notebook should have three main #header sections:
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
