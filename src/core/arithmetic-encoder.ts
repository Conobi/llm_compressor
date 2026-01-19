import { BitOutputStream } from './bit-stream.js';
import { buildCumulativeDistribution, getSymbolRange } from './cumulative-dist.js';

/**
 * Constants for arithmetic coding.
 * Using 32 bits total for state precision.
 */
const HALF = 0x80000000; // 2^31
const QUARTER = 0x40000000; // 2^30
const THREE_QUARTERS = 0xc0000000; // 3 * 2^30
const MASK = 0xffffffff;

/**
 * Arithmetic encoder using 32-bit precision with renormalization.
 */
export class ArithmeticEncoder {
  private low: number = 0;
  private high: number = MASK; // 0xFFFFFFFF
  private pendingBits: number = 0;
  private output: BitOutputStream;

  constructor(output: BitOutputStream) {
    this.output = output;
  }

  /**
   * Encode a symbol given its probability distribution.
   */
  encode(symbol: number, probabilities: Float32Array): void {
    const cumulative = buildCumulativeDistribution(probabilities);
    this.encodeWithCumulative(symbol, cumulative);
  }

  /**
   * Encode a symbol using a pre-computed cumulative distribution.
   */
  encodeWithCumulative(symbol: number, cumulative: Uint32Array): void {
    const { low: symLow, high: symHigh, total } = getSymbolRange(
      cumulative,
      symbol
    );

    // Compute new interval using BigInt for intermediate calculations
    const range = BigInt((this.high >>> 0) - (this.low >>> 0) + 1);
    const newLow =
      (this.low >>> 0) + Number((range * BigInt(symLow)) / BigInt(total));
    const newHigh =
      (this.low >>> 0) +
      Number((range * BigInt(symHigh)) / BigInt(total)) -
      1;

    this.low = newLow >>> 0;
    this.high = newHigh >>> 0;

    // Renormalization
    while (true) {
      if ((this.high >>> 0) < HALF) {
        // MSB of both is 0
        this.writeBitPlusPending(0);
      } else if ((this.low >>> 0) >= HALF) {
        // MSB of both is 1
        this.writeBitPlusPending(1);
        this.low = (this.low - HALF) >>> 0;
        this.high = (this.high - HALF) >>> 0;
      } else if ((this.low >>> 0) >= QUARTER && (this.high >>> 0) < THREE_QUARTERS) {
        // Second MSB differs, center the range
        this.pendingBits++;
        this.low = (this.low - QUARTER) >>> 0;
        this.high = (this.high - QUARTER) >>> 0;
      } else {
        break;
      }

      // Double the range
      this.low = (this.low << 1) >>> 0;
      this.high = ((this.high << 1) | 1) >>> 0;
    }
  }

  private writeBitPlusPending(bit: number): void {
    this.output.writeBit(bit);
    while (this.pendingBits > 0) {
      this.output.writeBit(bit ^ 1);
      this.pendingBits--;
    }
  }

  /**
   * Finalize encoding.
   */
  finish(): void {
    this.pendingBits++;
    if ((this.low >>> 0) < QUARTER) {
      this.writeBitPlusPending(0);
    } else {
      this.writeBitPlusPending(1);
    }
    this.output.flush();
  }

  /**
   * Reset encoder state.
   */
  reset(): void {
    this.low = 0;
    this.high = MASK;
    this.pendingBits = 0;
  }
}
