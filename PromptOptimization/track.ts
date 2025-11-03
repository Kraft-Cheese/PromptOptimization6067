export class ProgressTracker {
  private startTime: number;
  private totalSteps: number;
  private currentStep: number;

  constructor(totalSteps: number) {
    this.startTime = Date.now();
    this.totalSteps = totalSteps;
    this.currentStep = 0;
  }

  update(step?: number) {
    if (step !== undefined) {
      this.currentStep = step;
    } else {
      this.currentStep++;
    }

    const elapsed = (Date.now() - this.startTime) / 1000;
    const progress = (this.currentStep / this.totalSteps) * 100;
    const eta =
      this.currentStep > 0
        ? ((elapsed / this.currentStep) * (this.totalSteps - this.currentStep))
        : 0;

    console.log(
      `Progress: ${this.currentStep}/${this.totalSteps} (${progress.toFixed(1)}%) | ` +
      `Elapsed: ${elapsed.toFixed(1)}s | ETA: ${eta.toFixed(1)}s`
    );
  }
}