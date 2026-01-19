/**
 * Utilities for converting probability distributions to cumulative distributions
 * suitable for arithmetic coding.
 *
 * We use 16-bit precision for probability scaling to ensure safe integer math
 * when combined with the 32-bit interval range.
 */

/**
 * Scale factor for converting probabilities to integers.
 * Using 16 bits (65536) for probability precision.
 */
export const PROB_SCALE = 65536;

/**
 * Minimum count for any symbol (prevents zero-probability issues).
 */
export const MIN_COUNT = 1;

/**
 * Convert a probability distribution to a cumulative distribution.
 *
 * @param probs - Float32Array of probabilities (should sum to ~1.0)
 * @returns Uint32Array of cumulative values, length = probs.length + 1
 *          cumulative[i] = sum of scaled probabilities for symbols 0..i-1
 *          cumulative[0] = 0
 *          cumulative[n] = total (approximately PROB_SCALE)
 */
export function buildCumulativeDistribution(probs: Float32Array): Uint32Array {
  const n = probs.length;
  const cumulative = new Uint32Array(n + 1);
  cumulative[0] = 0;

  // First pass: compute scaled counts
  const counts = new Uint32Array(n);
  let total = 0;

  for (let i = 0; i < n; i++) {
    // Scale probability to integer, ensuring minimum of MIN_COUNT
    counts[i] = Math.max(MIN_COUNT, Math.floor(probs[i] * PROB_SCALE));
    total += counts[i];
  }

  // Build cumulative distribution
  let sum = 0;
  for (let i = 0; i < n; i++) {
    cumulative[i] = sum;
    sum += counts[i];
  }
  cumulative[n] = sum;

  return cumulative;
}

/**
 * Find which symbol a value falls into given cumulative distribution.
 * Uses binary search for efficiency with large vocabularies.
 *
 * @param cumulative - Cumulative distribution from buildCumulativeDistribution
 * @param target - Scaled value to find (0 <= target < cumulative[n])
 * @returns Symbol index (0 to n-1)
 */
export function findSymbol(cumulative: Uint32Array, target: number): number {
  // Binary search for largest i where cumulative[i] <= target
  let lo = 0;
  let hi = cumulative.length - 2; // symbols are 0 to n-1

  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (cumulative[mid] <= target) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }

  return lo;
}

/**
 * Get the probability range for a specific symbol.
 *
 * @param cumulative - Cumulative distribution
 * @param symbol - Symbol index
 * @returns [low, high) range where low = cumulative[symbol], high = cumulative[symbol+1]
 */
export function getSymbolRange(
  cumulative: Uint32Array,
  symbol: number
): { low: number; high: number; total: number } {
  return {
    low: cumulative[symbol],
    high: cumulative[symbol + 1],
    total: cumulative[cumulative.length - 1],
  };
}
