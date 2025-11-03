export class DummyEvaluator<E> {
  private armProbabilities: Map<string, number>;

  constructor(numArms: number) {
    this.armProbabilities = new Map();
    // Assign random probabilities to each arm
    for (let i = 0; i < numArms; i++) {
      this.armProbabilities.set(`arm_${i}`, Math.random());
    }
  }

  async evaluate(
    instruction: string,
    example: E
  ): Promise<{ score: number; tokens: number }> {
    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 10));

    // Extract arm ID or use instruction hash
    const armId = this.getArmId(instruction);
    const baseProb = this.armProbabilities.get(armId) ?? 0.5;

    // Add noise
    const noise = (Math.random() - 0.5) * 0.2;
    const score = Math.max(0, Math.min(1, baseProb + noise));

    // Dummy token count
    const tokens = Math.floor(Math.random() * 100) + 50;

    return { score, tokens };
  }

  private getArmId(instruction: string): string {
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < instruction.length; i++) {
      hash = (hash << 5) - hash + instruction.charCodeAt(i);
      hash = hash & hash;
    }
    const armIndex = Math.abs(hash) % this.armProbabilities.size;
    return `arm_${armIndex}`;
  }
}