# Agent Instructions

This repository is a Three.js project utilizing WebGPU, TSL (Three.js Shading Language), and compute shaders to perform ocean simulation (JONSWAP spectrum and IFFT).

## Commands

- `npm run dev`: Starts the local development server (Webpack Dev Server).
- `npm run build`: Bundles the project for production into the `dist/` directory.

## Architecture & Entrypoints

- `src/script.ts`: The main application entrypoint. It initializes the `WebGPURenderer`, scene, GUI controls, and handles the compute shader cascade orchestration.
- `bundler/`: Contains Webpack configurations (`webpack.dev.js`, `webpack.prod.js`, `webpack.common.js`).
- `src/shaders/` & `src/simulation/`: Contains logic for TSL nodes and WebGPU compute shaders.
- Styling is handled by `src/style.css` which is imported directly into `script.ts`.

## Framework & Toolchain Quirks

- **WebGPU Only**: The project specifically imports from `three/webgpu` instead of standard `three`. Standard materials (like `MeshStandardMaterial`) are replaced by their Node variants (e.g., `MeshBasicNodeMaterial`).
- **Compute Shaders**: Compute operations (e.g., updating spectrum and running IFFT passes) must be executed explicitly inside the active render loop using `renderer.compute(...)`.
- **TypeScript**: Configured with `"moduleResolution": "bundler"` and `"target": "ESNext"`. Imports for WebGPU submodules and `.css` files are standard.
