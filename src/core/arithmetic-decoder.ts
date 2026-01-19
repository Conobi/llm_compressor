import { BitInputStream } from './bit-stream.js';
import {
  buildCumulativeDistribution,
  findSymbol,
  getSymbolRange,
} from './cumulative-dist.js';

/**
 * Constants for arithmetic coding (must match encoder).
 */
const STATE_BITS = 32;
const HALF = 0x80000000; // 2^31
const QUARTER = 0x40000000; // 2^30
const THREE_QUARTERS = 0xc0000000; // 3 * 2^30
const MASK = 0xffffffff;

/**
 * Arithmetic decoder using 32-bit precision with renormalization.
 */
export class ArithmeticDecoder {
  private low: number = 0;
  private high: number = MASK;
  private code: number = 0;
  private input: BitInputStream;

  constructor(input: BitInputStream) {
    this.input = input;
    // Initialize code with first 32 bits
    for (let i = 0; i < STATE_BITS; i++) {
      this.code = ((this.code << 1) | this.input.readBit()) >>> 0;
    }
  }

  /**
   * Decode a symbol given its probability distribution.
   */
  decode(probabilities: Float32Array): number {
    const cumulative = buildCumulativeDistribution(probabilities);
    return this.decodeWithCumulative(cumulative);
  }

  /**
   * Decode a symbol using a pre-computed cumulative distribution.
   */
  decodeWithCumulative(cumulative: Uint32Array): number {
    const total = cumulative[cumulative.length - 1];

    // Compute range using BigInt
    const range = BigInt((this.high >>> 0) - (this.low >>> 0) + 1);

    // Find the symbol
    // offset = code - low
    // scaled = floor((offset + 1) * total - 1) / range)
    const offset = BigInt((this.code >>> 0) - (this.low >>> 0));
    const scaled = Number(((offset + 1n) * BigInt(total) - 1n) / range);

    const symbol = findSymbol(cumulative, scaled);

    // Update interval (same as encoder)
    const { low: symLow, high: symHigh } = getSymbolRange(cumulative, symbol);
    const newLow =
      (this.low >>> 0) + Number((range * BigInt(symLow)) / BigInt(total));
    const newHigh =
      (this.low >>> 0) +
      Number((range * BigInt(symHigh)) / BigInt(total)) -
      1;

    this.low = newLow >>> 0;
    this.high = newHigh >>> 0;

    // Renormalization (must match encoder exactly)
    while (true) {
      if ((this.high >>> 0) < HALF) {
        // MSB of both is 0 - do nothing special
      } else if ((this.low >>> 0) >= HALF) {
        // MSB of both is 1
        this.code = (this.code - HALF) >>> 0;
        this.low = (this.low - HALF) >>> 0;
        this.high = (this.high - HALF) >>> 0;
      } else if (
        (this.low >>> 0) >= QUARTER &&
        (this.high >>> 0) < THREE_QUARTERS
      ) {
        // Second MSB differs
        this.code = (this.code - QUARTER) >>> 0;
        this.low = (this.low - QUARTER) >>> 0;
        this.high = (this.high - QUARTER) >>> 0;
      } else {
        break;
      }

      // Double the range and read next bit
      this.low = (this.low << 1) >>> 0;
      this.high = ((this.high << 1) | 1) >>> 0;
      this.code = ((this.code << 1) | this.input.readBit()) >>> 0;
    }

    return symbol;
  }

  /**
   * Reset decoder state.
   */
  reset(input: BitInputStream): void {
    this.input = input;
    this.low = 0;
    this.high = MASK;
    this.code = 0;

    for (let i = 0; i < STATE_BITS; i++) {
      this.code = ((this.code << 1) | this.input.readBit()) >>> 0;
    }
  }
}
