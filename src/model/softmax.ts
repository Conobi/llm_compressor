/**
 * Numerically stable softmax implementation.
 *
 * Converts logits (raw model outputs) to probabilities.
 */

/**
 * Compute softmax of logits.
 *
 * Uses the max-subtraction trick for numerical stability:
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * @param logits - Raw model output values
 * @returns Float32Array of probabilities (sums to 1.0)
 */
export function softmax(logits: Float32Array | number[]): Float32Array {
  const n = logits.length;
  const result = new Float32Array(n);

  // Find maximum for numerical stability
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }

  // Compute exp(x - max) and sum
  let sum = 0;
  for (let i = 0; i < n; i++) {
    result[i] = Math.exp(logits[i] - max);
    sum += result[i];
  }

  // Normalize to probabilities
  const invSum = 1 / sum;
  for (let i = 0; i < n; i++) {
    result[i] *= invSum;
  }

  return result;
}

/**
 * Compute log-softmax (more numerically stable for log probabilities).
 *
 * @param logits - Raw model output values
 * @returns Float32Array of log probabilities
 */
export function logSoftmax(logits: Float32Array | number[]): Float32Array {
  const n = logits.length;
  const result = new Float32Array(n);

  // Find maximum
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }

  // Compute log-sum-exp
  let sumExp = 0;
  for (let i = 0; i < n; i++) {
    sumExp += Math.exp(logits[i] - max);
  }
  const logSumExp = max + Math.log(sumExp);

  // Compute log probabilities
  for (let i = 0; i < n; i++) {
    result[i] = logits[i] - logSumExp;
  }

  return result;
}
