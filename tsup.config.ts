import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  minify: false,
  // Mark node built-ins and onnx runtime as external
  external: ['onnxruntime-web', 'onnxruntime-node', 'fs', 'fs/promises', 'os'],
  noExternal: [],
  esbuildOptions(options) {
    options.platform = 'neutral';
  },
});
