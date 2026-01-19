# notebox-compressor

LLM-based lossless text compression using RWKV and arithmetic coding.

This module achieves high compression ratios by using a language model to predict token probabilities, then encoding with arithmetic coding. Based on the approach from [Bellard's ts_zip](https://bellard.org/ts_zip/) and the "Language Modeling Is Compression" paper.

## Features

- Lossless text compression with ~3-4x better ratios than gzip
- Works in both Node.js and browser environments
- Uses RWKV-4-pile-169m model via ONNX Runtime
- Pure JavaScript BPE tokenizer (GPT-NeoX vocabulary)

## Installation

```bash
pnpm install
```

## Download Model

Download the RWKV model (~170MB):

```bash
curl -L https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx \
  -o ./assets/rwkv-4-pile-169m-uint8.onnx
```

## Usage

### Command Line Example

```bash
# Compress a string
pnpm exec tsx examples/compress.ts "Hello, world!"

# Compress a file
pnpm exec tsx examples/compress.ts "$(cat yourfile.txt)"
```

### Programmatic Usage

```typescript
import { LLMCompressor } from 'notebox-compressor';

const compressor = new LLMCompressor({
  model: './assets/rwkv-4-pile-169m-uint8.onnx',
  tokenizer: './assets/20B_tokenizer.json',
});

await compressor.init();

// Compress
const result = await compressor.compress('Hello, world!');
console.log(`Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
console.log(`Compressed size: ${result.compressedSize} bytes`);

// Decompress
const text = await compressor.decompress(result.data);
console.log(text); // 'Hello, world!'

compressor.dispose();
```

## API

### `LLMCompressor`

#### Constructor Options

```typescript
interface CompressorOptions {
  model: string | ArrayBuffer;    // Path/URL to ONNX model or ArrayBuffer
  tokenizer: string | ArrayBuffer; // Path/URL to tokenizer JSON or ArrayBuffer
  wasmThreads?: number;           // Number of WASM threads (default: auto)
  onProgress?: (info: ProgressInfo) => void; // Progress callback
}
```

#### Methods

- `init(): Promise<void>` - Initialize the model and tokenizer
- `compress(text: string): Promise<CompressionResult>` - Compress text
- `decompress(data: Uint8Array): Promise<string>` - Decompress data
- `dispose(): void` - Release resources

#### CompressionResult

```typescript
interface CompressionResult {
  data: Uint8Array;        // Compressed data
  originalSize: number;    // Original size in bytes
  compressedSize: number;  // Compressed size in bytes
  compressionRatio: number; // originalSize / compressedSize
  tokenCount: number;      // Number of tokens
}
```

## Development

```bash
# Build
pnpm build

# Run tests
pnpm test

# Type check
pnpm typecheck
```

## Performance

- Compression ratio: ~0.7-1.5 bits/character (vs gzip's 2-3 bits/character)
- Speed: ~50-100ms per token (WASM backend)
- Memory: ~500MB peak during inference

## Limitations

- Slow compression/decompression (LLM inference per token)
- Requires same model for compression and decompression
- Large model file (~170MB)

## License

MIT
