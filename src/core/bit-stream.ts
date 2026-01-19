/**
 * Bit-level output stream for arithmetic coding.
 * Accumulates bits and outputs bytes when full.
 */
export class BitOutputStream {
  private buffer: number[] = [];
  private currentByte: number = 0;
  private bitPosition: number = 0;

  /**
   * Write a single bit to the stream.
   * @param bit - 0 or 1
   */
  writeBit(bit: number): void {
    this.currentByte = (this.currentByte << 1) | (bit & 1);
    this.bitPosition++;

    if (this.bitPosition === 8) {
      this.buffer.push(this.currentByte);
      this.currentByte = 0;
      this.bitPosition = 0;
    }
  }

  /**
   * Write multiple bits from a number (MSB first).
   * @param value - The value containing the bits
   * @param count - Number of bits to write (1-32)
   */
  writeBits(value: number, count: number): void {
    for (let i = count - 1; i >= 0; i--) {
      this.writeBit((value >>> i) & 1);
    }
  }

  /**
   * Flush any remaining bits, padding with zeros.
   * Must be called after all data is written.
   */
  flush(): void {
    if (this.bitPosition > 0) {
      this.currentByte <<= 8 - this.bitPosition;
      this.buffer.push(this.currentByte);
      this.currentByte = 0;
      this.bitPosition = 0;
    }
  }

  /**
   * Get the current byte count (before flush).
   */
  get byteCount(): number {
    return this.buffer.length;
  }

  /**
   * Get the total bit count written.
   */
  get bitCount(): number {
    return this.buffer.length * 8 + this.bitPosition;
  }

  /**
   * Convert the stream to a Uint8Array.
   * Call flush() first if you want to include partial bytes.
   */
  toUint8Array(): Uint8Array {
    return new Uint8Array(this.buffer);
  }

  /**
   * Extract accumulated bytes and reset buffer.
   * Useful for streaming output.
   */
  extractBytes(): Uint8Array {
    const result = new Uint8Array(this.buffer);
    this.buffer = [];
    return result;
  }
}

/**
 * Bit-level input stream for arithmetic decoding.
 * Reads bits from a Uint8Array.
 */
export class BitInputStream {
  private data: Uint8Array;
  private bytePosition: number = 0;
  private bitPosition: number = 0;

  constructor(data: Uint8Array) {
    this.data = data;
  }

  /**
   * Read a single bit from the stream.
   * Returns 0 when past end of data (padding behavior).
   */
  readBit(): number {
    if (this.bytePosition >= this.data.length) {
      return 0; // Pad with zeros at end
    }

    const bit = (this.data[this.bytePosition] >>> (7 - this.bitPosition)) & 1;
    this.bitPosition++;

    if (this.bitPosition === 8) {
      this.bytePosition++;
      this.bitPosition = 0;
    }

    return bit;
  }

  /**
   * Read multiple bits as a number (MSB first).
   * @param count - Number of bits to read (1-32)
   */
  readBits(count: number): number {
    let value = 0;
    for (let i = 0; i < count; i++) {
      value = (value << 1) | this.readBit();
    }
    return value;
  }

  /**
   * Check if we've reached the end of the data.
   */
  get isAtEnd(): boolean {
    return this.bytePosition >= this.data.length;
  }

  /**
   * Get current position in bits.
   */
  get position(): number {
    return this.bytePosition * 8 + this.bitPosition;
  }

  /**
   * Get total size in bits.
   */
  get size(): number {
    return this.data.length * 8;
  }
}
