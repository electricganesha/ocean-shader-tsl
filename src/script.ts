import "./style.css";
import * as THREE from "three";
import { WebGPURenderer, MeshBasicNodeMaterial, MeshStandardNodeMaterial } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import * as dat from "lil-gui";
import {
  WaterNodeMaterial,
  WaterMaterialParams,
} from "./shaders/water/WaterNodeMaterial";
import { createJonswapCompute } from "./simulation/JonswapSpectrum";
import { createIFFTCompute } from "./simulation/IFFTCompute";

/**
 * Base
 */
// Debug
const gui = new dat.GUI({ width: 340 });
gui.close();
interface DebugObject {
  depthColor: string;
  surfaceColor: string;
  timeScale: number;
  enableSSS: boolean;
  enableFoam: boolean;
  enableWaves: boolean;
  windDirX: number;
  windDirZ: number;
  sunAzimuth: number;
  sunElevation: number;
}
const debugObject: DebugObject = {
  depthColor: "#050c14",
  surfaceColor: "#0a1c2e",
  timeScale: 0.65,
  enableSSS: true,
  enableFoam: true,
  enableWaves: true,
  windDirX: 1.0,
  windDirZ: 0.0,
  sunAzimuth: 45.0,
  sunElevation: 40.0,
};

// Canvas
const canvas = document.querySelector("canvas.webgl") as HTMLCanvasElement;

const textureLoader = new THREE.TextureLoader();
const normalMap1 = textureLoader.load("/textures/water/normal1.png");
const normalMap2 = textureLoader.load("/textures/water/normal2.png");
const foamMap = textureLoader.load("/textures/water/foam.png");
normalMap1.wrapS = normalMap1.wrapT = THREE.RepeatWrapping;
normalMap2.wrapS = normalMap2.wrapT = THREE.RepeatWrapping;
foamMap.wrapS = foamMap.wrapT = THREE.RepeatWrapping;

/**
 * Sizes
 */
interface Sizes {
  width: number;
  height: number;
}
const sizes: Sizes = {
  width: window.innerWidth,
  height: window.innerHeight,
};

/**
 * Renderer
 */
const renderer = new WebGPURenderer({
  canvas: canvas,
  antialias: true,
});
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// Water Simulation Cascades
const size = 256;
const baseL = 25; // Adjusted from 100 to 25 to match the 10m diorama box scale
const phi = 1.618; // Golden ratio for cascade scaling

// Cascade 1: Large scale waves
const cascade1 = createJonswapCompute(size, baseL);
cascade1.uniforms.kMin.value = 0;
cascade1.uniforms.kMax.value = 10.0; // Wide range to cover all frequencies in one cascade for the 10m diorama box

// Cascade 2: Mid scale waves (Golden Ratio scale)
const cascade2 = createJonswapCompute(size, baseL * phi);
cascade2.uniforms.kMin.value = 0.5;
cascade2.uniforms.kMax.value = 1.5;

// Cascade 3: Small detail ripples (Golden Ratio squared scale)
const cascade3 = createJonswapCompute(size, baseL * phi * phi);
cascade3.uniforms.kMin.value = 1.5;
cascade3.uniforms.kMax.value = 10.0;

const cascades = [cascade1, cascade2, cascade3];

// IFFT for Cascade 1
const ifftVertical = createIFFTCompute(size, cascade1.timeEvolvedStorage);
const ifftHorizontal = createIFFTCompute(size, cascade1.horizontalStorage);
const ifftSlopes = createIFFTCompute(size, cascade1.slopesStorage);
const ifftJacobian1 = createIFFTCompute(size, cascade1.jacobianStorage1);
const ifftJacobian2 = createIFFTCompute(size, cascade1.jacobianStorage2);

// Scene
const scene = new THREE.Scene();
const cubeTextureLoader = new THREE.CubeTextureLoader();
const environmentMap = cubeTextureLoader.load([
  "/environmentMaps/1/px.png",
  "/environmentMaps/1/nx.png",
  "/environmentMaps/1/py.png",
  "/environmentMaps/1/ny.png",
  "/environmentMaps/1/pz.png",
  "/environmentMaps/1/nz.png",
]);
scene.background = environmentMap;

// Global Lighting (Required for MeshStandardNodeMaterial)
const ambientLight = new THREE.AmbientLight("#ffffff", 1.5);
scene.add(ambientLight);

const hemisphereLight = new THREE.HemisphereLight("#ffffff", "#000000", 1.0);
scene.add(hemisphereLight);

const sunLight = new THREE.DirectionalLight("#ffffff", 4.0);
scene.add(sunLight);

/**
 * Water
 */
// Geometry
const waterGeometry = new THREE.BoxGeometry(10, 2, 10, 512, 1, 512);

// Material (TSL Nodes)
const waterNodeData = WaterNodeMaterial({
  envMap: environmentMap,
  normalMap1: normalMap1,
  normalMap2: normalMap2,
  foamMap: foamMap,
  vDisp1: ifftVertical.outputStorage,
  hDisp1: ifftHorizontal.outputStorage,
  vSlopeX: ifftSlopes.outputStorage,
  vSlopeZ: ifftSlopes.outputStorage,
  j1: ifftJacobian1.outputStorage,
  j2: ifftJacobian2.outputStorage,
  gridSize: size,
  simulationL: baseL, // Must match baseL — keeps UV wrapping in sync with simulation domain
} as WaterMaterialParams);

const waterMaterial = new MeshBasicNodeMaterial();
waterMaterial.positionNode = waterNodeData.positionNode;
waterMaterial.colorNode = waterNodeData.colorNode;
waterMaterial.transparent = waterNodeData.transparent;
waterMaterial.opacityNode = waterNodeData.opacity;
waterMaterial.side = THREE.DoubleSide;

const water = new THREE.Mesh(waterGeometry, waterMaterial);
// Must render AFTER the environment so viewportSharedTexture has scene content
water.renderOrder = 1;
scene.add(water);

/**
 * Buoy (Loaded OBJ Model)
 */
const buoyGroup = new THREE.Group();
scene.add(buoyGroup);

// Point Light for the beacon (keep this as a helper)
const buoyLight = new THREE.PointLight("#ffaa00", 12, 4);
buoyLight.position.set(0, 1.45, 0); // Default position, will be updated if needed
buoyGroup.add(buoyLight);

// Load Model and Texture
const objLoader = new OBJLoader();
const buoyTexture = textureLoader.load("/models/buoy.png");
buoyTexture.colorSpace = THREE.SRGBColorSpace;

objLoader.load("/models/buoy.obj", (obj) => {
  const buoyMaterial = new MeshStandardNodeMaterial({
    map: buoyTexture,
    roughness: 0.4,
    metalness: 0.3,
    envMap: environmentMap, // Add reflections for realism
  });

  obj.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      (child as THREE.Mesh).material = buoyMaterial;
    }
  });

  // Compute the bounding box to center it correctly
  const box = new THREE.Box3().setFromObject(obj);
  const size = new THREE.Vector3();
  box.getSize(size);
  
  const center = new THREE.Vector3();
  box.getCenter(center);

  // Center horizontally
  obj.position.x = -center.x;
  obj.position.z = -center.z;
  
  // Align bottom of buoy to y=0 of the group (the water surface)
  // We'll push it down by about 4% of its height to look properly "weighted" and submerged.
  obj.position.y = -box.min.y - (size.y * 0.04); 

  obj.scale.setScalar(0.2); 
  
  buoyGroup.add(obj);
  
  // Position the light at the very top of the model
  const updatedBox = new THREE.Box3().setFromObject(obj);
  buoyLight.position.y = updatedBox.max.y;
});

/**
 * Buoyancy Physics (CPU side Gerstner matching)
 */
const getWaveHeight = (x: number, z: number, time: number) => {
  // Use the live uniform values from the material
  const waves = [
    waterNodeData.uniforms.uGerstnerWaveA.value,
    waterNodeData.uniforms.uGerstnerWaveB.value,
    waterNodeData.uniforms.uGerstnerWaveC.value,
    waterNodeData.uniforms.uGerstnerWaveD.value,
  ];

  let posY = 0;
  let dHdx = 0;
  let dHdz = 0;

  waves.forEach((w) => {
    const dir = new THREE.Vector2(w.x, w.y).normalize();
    const steepness = w.z;
    const wavelength = w.w;
    
    const k = (2 * Math.PI) / wavelength;
    const c = Math.sqrt(9.81 / k);
    const f = k * (dir.dot(new THREE.Vector2(x, z)) - c * time);
    
    posY += (steepness / k) * Math.sin(f);
    
    const d = steepness * Math.cos(f);
    dHdx += d * dir.x;
    dHdz += d * dir.y;
  });

  const normal = new THREE.Vector3(-dHdx, 1, -dHdz).normalize();
  return { y: posY, normal };
};

// ── GUI Controls ───────────────────────────────────────────────────────────
const updateSpectrum = () => {
  if ((window as any).__setSpectrumDirty) {
    (window as any).__setSpectrumDirty();
  }
};

const updateWindDir = () => {
  const len = Math.sqrt(debugObject.windDirX ** 2 + debugObject.windDirZ ** 2);
  const nx = len > 0.001 ? debugObject.windDirX / len : 1.0;
  const nz = len > 0.001 ? debugObject.windDirZ / len : 0.0;
  cascades.forEach((c) => {
    c.uniforms.windDir.value.set(nx, nz);
  });
  updateSpectrum();
};

const updateSunDirection = () => {
  const az = (debugObject.sunAzimuth * Math.PI) / 180;
  const el = (debugObject.sunElevation * Math.PI) / 180;
  
  const sunDir = new THREE.Vector3(
    Math.cos(el) * Math.sin(az),
    Math.sin(el),
    Math.cos(el) * Math.cos(az),
  );
  
  waterNodeData.uniforms.uSunDirection.value.copy(sunDir);
  
  // Also update the scene light for the buoy
  sunLight.position.copy(sunDir).multiplyScalar(10);
};

const simFolder = gui.addFolder("Ocean Simulation & Geometry");
simFolder.add(debugObject, "timeScale", 0, 5, 0.01).name("Time Scale");
simFolder.add(waterNodeData.uniforms.uChoppiness, "value", 0, 3, 0.05).name("Choppiness (λ)");

const jonswapFolder = simFolder.addFolder("JONSWAP Spectrum");
jonswapFolder.add(cascade1.uniforms.windSpeed, "value", 0, 50, 0.1).name("Wind Speed").onChange(() => updateSpectrum());
jonswapFolder.add(cascade1.uniforms.fetch, "value", 1000, 500000, 100).name("Fetch").onChange(() => updateSpectrum());
jonswapFolder.add(cascade1.uniforms.gamma, "value", 1.0, 10.0, 0.1).name("Gamma").onChange(() => updateSpectrum());
jonswapFolder.add(cascade1.uniforms.depth, "value", 1.0, 500.0, 1.0).name("Bottom Depth").onChange(() => updateSpectrum());
jonswapFolder.add(debugObject, "windDirX", -1, 1, 0.01).name("Wind Dir X").onChange(updateWindDir);
jonswapFolder.add(debugObject, "windDirZ", -1, 1, 0.01).name("Wind Dir Z").onChange(updateWindDir);

const gerstnerFolder = simFolder.addFolder("Gerstner Waves (Geometry)");
const addWaveControls = (folder: dat.GUI, uniform: any, name: string) => {
  const f = folder.addFolder(name);
  f.add(uniform.value, "x", -1, 1, 0.01).name("Dir X");
  f.add(uniform.value, "y", -1, 1, 0.01).name("Dir Z");
  f.add(uniform.value, "z", 0, 1, 0.01).name("Steepness");
  f.add(uniform.value, "w", 0.1, 10, 0.01).name("Wavelength");
};
addWaveControls(gerstnerFolder, waterNodeData.uniforms.uGerstnerWaveA, "Wave A");
addWaveControls(gerstnerFolder, waterNodeData.uniforms.uGerstnerWaveB, "Wave B");
addWaveControls(gerstnerFolder, waterNodeData.uniforms.uGerstnerWaveC, "Wave C");
addWaveControls(gerstnerFolder, waterNodeData.uniforms.uGerstnerWaveD, "Wave D");

const lightingFolder = gui.addFolder("Atmosphere & Lighting");
lightingFolder.add(debugObject, "sunAzimuth", 0, 360, 1).name("Sun Azimuth (°)").onChange(updateSunDirection);
lightingFolder.add(debugObject, "sunElevation", 0, 90, 1).name("Sun Elevation (°)").onChange(updateSunDirection);
lightingFolder.addColor({ color: "#fff5e0" }, "color").name("Sun Color").onChange((v: string) => waterNodeData.uniforms.uSunColor.value.set(v));
lightingFolder.add(waterNodeData.uniforms.uSunSpecularPower, "value", 10, 3000, 10).name("Glint Sharpness");
lightingFolder.add(waterNodeData.uniforms.uSunSpecularStrength, "value", 0, 50, 0.1).name("Glint Strength");
lightingFolder.add(debugObject, "enableWaves", 0.0, 1.0, 1.0).name("Enable Wave Geometry").onChange((v: any) => waterNodeData.uniforms.uEnableWaves.value = v);

const opticsFolder = gui.addFolder("Optical Properties");
opticsFolder.add(waterNodeData.uniforms.uIOR, "value", 1.0, 1.5, 0.001).name("IOR (Refraction)");
opticsFolder.add(waterNodeData.uniforms.uRefrStrength, "value", 0, 0.1, 0.001).name("Refraction Distortion");
opticsFolder.add(waterNodeData.uniforms.uDispersion, "value", 0, 0.02, 0.0001).name("Chromatic Dispersion");
opticsFolder.add(waterNodeData.uniforms.uDepthScale, "value", 0.1, 50, 0.1).name("Optical Density (m)");
opticsFolder.add(waterNodeData.uniforms.uOpacity, "value", 0.0, 1.0, 0.01).name("Global Opacity");

const featuresFolder = gui.addFolder("Surface Features");
const sssFolder = featuresFolder.addFolder("Subsurface Scattering (SSS)");
sssFolder.add(debugObject, "enableSSS").name("Enable").onChange((v: boolean) => waterNodeData.uniforms.uEnableSSS.value = v ? 1.0 : 0.0);
sssFolder.addColor({ color: "#00d4ff" }, "color").name("Tint").onChange((v: string) => waterNodeData.uniforms.uSSSColor.value.set(v));
sssFolder.add(waterNodeData.uniforms.uSSSStrength, "value", 0, 5, 0.05).name("Strength");

const foamFolder = featuresFolder.addFolder("Breaking Foam");
foamFolder.add(debugObject, "enableFoam").name("Enable").onChange((v: boolean) => waterNodeData.uniforms.uEnableFoam.value = v ? 1.0 : 0.0);
foamFolder.add(waterNodeData.uniforms.uFoamThreshold, "value", 0, 2, 0.01).name("Break Threshold");
foamFolder.addColor({ color: "#e8f4f8" }, "color").name("Color").onChange((v: string) => waterNodeData.uniforms.uFoamColor.value.set(v));

const causticsFolder = featuresFolder.addFolder("Analytic Caustics");
causticsFolder.add(waterNodeData.uniforms.uEnableCaustics, "value", 0, 1, 1).name("Enable");
causticsFolder.add(waterNodeData.uniforms.uCausticsStrength, "value", 0, 5, 0.05).name("Strength");
causticsFolder.add(waterNodeData.uniforms.uCausticsSharpness, "value", 1, 12, 0.1).name("Sharpness");

const appearanceFolder = gui.addFolder("Water Appearance");
appearanceFolder.addColor(debugObject, "surfaceColor").name("Base Tint").onChange(() => waterNodeData.uniforms.uSurfaceColor.value.set(debugObject.surfaceColor));
appearanceFolder.add(waterNodeData.uniforms.uUseTextures, "value").name("Normal Textures");
appearanceFolder.add(waterNodeData.uniforms.uNormalIntensity, "value", 0, 1, 0.01).name("Micro Ripples Intensity");
appearanceFolder.add(waterNodeData.uniforms.uNormalScale, "value", 0, 20, 0.1).name("Micro Ripples Scale");

window.addEventListener("resize", () => {
  // Update sizes
  sizes.width = window.innerWidth;
  sizes.height = window.innerHeight;

  // Update camera
  camera.aspect = sizes.width / sizes.height;
  camera.updateProjectionMatrix();

  // Update renderer
  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(
  75,
  sizes.width / sizes.height,
  0.1,
  100,
);
camera.position.set(8, 8, 8);
scene.add(camera);

// Controls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
// controls.maxPolarAngle = Math.PI / 2.2;
controls.enableRotate = true;
controls.enableZoom = true;

camera.position.set(0, 3, 10);
controls.update();

// Initialize Renderer and Start Loop (Async)
const initWebGPU = async () => {
  // Diagnostic
  if (navigator.gpu) {
    console.log("WebGPU: Supported in navigator.gpu");
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      console.log("WebGPU: Adapter acquired", adapter);
    } else {
      console.warn(
        "WebGPU: navigator.gpu exists but no adapter found. Possible driver or hardware limitation.",
      );
    }
  } else {
    console.warn(
      "WebGPU: navigator.gpu is undefined. Check flags or browser version.",
    );
  }

  await renderer.init();
  console.log("Renderer Backend:", (renderer.backend as any).constructor.name);

  // Dirty flag: set when JONSWAP parameters change, consumed at the start of the next GPU frame
  let spectrumDirty = true; // start dirty so the initial spectrum runs inside the WebGPU loop

  let lastTime = performance.now();
  let accumulatedTime = 0;

  // Update updateSpectrum to just set the flag — actual compute happens inside tick()
  // (overwrite the outer updateSpectrum to avoid the race condition)
  (window as any).__setSpectrumDirty = () => {
    spectrumDirty = true;
  };

  // Initialize lights and wind
  updateSunDirection();
  updateWindDir();

  const tick = () => {
    const currentTime = performance.now();
    const deltaTime = (currentTime - lastTime) / 1000;
    lastTime = currentTime;

    accumulatedTime += deltaTime * debugObject.timeScale;

    // Update controls
    controls.update();

    // Regenerate spectrum inside the GPU frame if parameters changed
    if (spectrumDirty) {
      spectrumDirty = false;
      for (const cascade of cascades) {
        renderer.compute(cascade.computeSpectrum);
      }
    }

    // Evolve cascades over time
    for (const cascade of cascades) {
      cascade.uniforms.uTime.value = accumulatedTime;
      renderer.compute(cascade.computeTimeEvolution);
    }

    // Update shader time
    waterNodeData.uniforms.uTime.value = accumulatedTime;

    // Update Buoy
    // Ensure buoy stays exactly in the center of the volume (X=0, Z=0)
    // The water surface baseline is at y=1.0 (top of the 2m box geometry)
    const buoyPos = getWaveHeight(0, 0, accumulatedTime);
    buoyGroup.position.set(0, 1.0 + buoyPos.y, 0);
    
    // Smoothly tilt buoy to match surface normal
    const targetQuaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      buoyPos.normal
    );
    buoyGroup.quaternion.slerp(targetQuaternion, 0.1);

    // Execute IFFT Passes for Cascade 1
    const runIFFT = (ifft: any) => {
      for (const pass of ifft.passes) {
        if (pass.step !== undefined) {
          ifft.uStep.value = pass.step;
        }
        renderer.compute(pass.compute);
      }
    };

    runIFFT(ifftVertical);
    runIFFT(ifftHorizontal);
    runIFFT(ifftSlopes);
    runIFFT(ifftJacobian1);
    runIFFT(ifftJacobian2);

    // Render
    renderer.render(scene, camera);

    // Call tick again on the next frame
    window.requestAnimationFrame(tick);
  };

  tick();
};

initWebGPU();
