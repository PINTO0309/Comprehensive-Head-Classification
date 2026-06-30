export {};

declare global {
  interface Window {
    electronBenchmark?: {
      finishCliBenchmark(payload: unknown): void;
    };
  }
}
