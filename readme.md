# Three.js WebGPU Ocean Simulation

A Three.js project utilizing WebGPU, TSL (Three.js Shading Language), and compute shaders to perform ocean simulation based on the JONSWAP spectrum and Inverse Fast Fourier Transform (IFFT).

## Features

- **WebGPU Powered:** Built exclusively for WebGPU utilizing `three/webgpu`.
- **Compute Shaders:** Highly parallelized generation of the ocean surface using compute operations.
- **JONSWAP Spectrum:** Realistic ocean wave spectrum calculation.
- **IFFT:** Cascaded Inverse Fast Fourier Transforms for efficient wave height map generation.
- **Node Materials (TSL):** Modern Three.js Shading Language implementation.
- **Interactive:** Real-time parameter tweaking via `lil-gui`.

## Requirements

- A modern browser with WebGPU support (e.g., Chrome, Edge).
- Node.js installed on your system.

## Installation

Clone the repository and install the dependencies (only the first time):

```bash
npm install
```

## Running Locally

Start the local development server (Webpack Dev Server):

```bash
npm run dev
```

The application will be accessible in your browser (usually at `http://localhost:8080`).

## Building for Production

Bundle the project into the `dist/` directory for production deployment:

```bash
npm run build
```

## Technical Solutions Implemented

This project employs several advanced rendering and simulation techniques:

1. **WebGPU Compute Shaders (Simulation Pipeline):** The core wave simulation runs entirely on the GPU via compute shaders, ensuring high performance.
2. **Cascaded JONSWAP Spectrum:** Generates realistic wave heights in the frequency domain considering wind speed, fetch, depth, and directional spreading. It employs a multi-cascade system for varied Level of Detail (LOD).
3. **Cooley-Tukey IFFT in TSL:** A Radix-2 Inverse Fast Fourier Transform (IFFT) is fully implemented in TSL. It includes bit-reversal and ping-pong butterfly passes to convert the frequency-domain spectrum into spatial displacement maps, normal maps, and Jacobians.
4. **Three.js Shading Language (TSL):** Both the compute pipeline and the rendering materials (`MeshBasicNodeMaterial`) heavily utilize TSL nodes instead of raw GLSL/WGSL, enabling modular and type-safe shader authoring.
5. **Hybrid Displacement (Gerstner + IFFT):** The final wave surface is a composite of analytical Gerstner waves and the IFFT displacement maps. The Gerstner calculations are also mirrored on the CPU to achieve physically accurate buoyancy and normal-aligned tilting for interactive objects like the floating buoy.
6. **Advanced Ocean Optics:**
   - **Volumetric Absorption (Beer-Lambert):** Ray-marched volumetric scattering factoring in water depth and specific optical constituents (pure water, phytoplankton, CDOM).
   - **Screen-Space Refraction & Dispersion:** Implements Snell's Law to distort background pixels, dynamically sampling RGB channels at slightly different indices of refraction to simulate chromatic dispersion.
   - **Subsurface Scattering (SSS):** Approximates light diffusion through thinner wave crests using a directional lighting lobe.
   - **Analytic Caustics:** Utilizes the wave's Jacobian determinant (`det(J)`) to highlight regions where surface normals focus sunlight onto the sea floor.
7. **Dynamic Foam Generation:** Calculates foam emergence mathematically using the Jacobian determinant (`det(J) < 1.0` denotes pinching/breaking wave crests) integrated with tiling noise textures for organic boundaries.
8. **LEADR Mapping (Anti-Aliasing):** Modifies the specular roughness based on distance and slope variance, suppressing specular aliasing in high-frequency distant waves.

## Architecture & Project Structure

- `src/script.ts`: The main application entrypoint. It initializes the `WebGPURenderer`, scene, GUI controls, and handles the compute shader cascade orchestration.
- `src/shaders/` & `src/simulation/`: Contains logic for TSL nodes and WebGPU compute shaders.
- `bundler/`: Contains Webpack configurations (`webpack.dev.js`, `webpack.prod.js`, `webpack.common.js`).
- `src/style.css`: Main styling for the application.

## Framework Notes

- **Imports**: The project specifically imports from `three/webgpu` instead of standard `three`. Standard materials are replaced by their Node variants (e.g., `MeshBasicNodeMaterial`).
- **Compute Execution**: Compute operations (like updating the spectrum and running IFFT passes) are executed explicitly inside the active render loop using `renderer.compute(...)`.
