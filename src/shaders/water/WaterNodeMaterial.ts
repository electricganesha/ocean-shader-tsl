import * as THREE from "three";
import {
  Fn,
  vec2,
  vec3,
  vec4,
  mat3,
  mul,
  add,
  sub,
  div,
  dot,
  cross,
  normalize,
  sin,
  cos,
  tan,
  sqrt,
  exp,
  pow,
  abs,
  floor,
  fract,
  clamp,
  mix,
  step,
  smoothstep,
  time,
  uv,
  positionLocal,
  positionWorld,
  normalLocal,
  normalWorld,
  normalView,
  cameraPosition,
  modelViewProjection,
  reflect,
  refract,
  normalize as normalizeNode,
  texture,
  cubeTexture,
  viewportUV,
  viewportSharedTexture,
  viewportDepthTexture,
  cameraNear,
  cameraFar,
  cameraProjectionMatrixInverse,
  cameraViewMatrix,
  property,
  uniform,
  varying,
  transformedNormalView,
  positionViewDirection,
  positionView,
  float,
  int,
  uint,
  bool,
  color,
  Discard,
  max,
  min,
  ceil,
  log2,
  Return,
  Loop,
} from "three/tsl";

/**
 * Continuous Distance-Based LOD Factor
 * Determines the subdivision/detail level based on distance from camera
 */
const getLODFactor = Fn(([posWorld]: [any]) => {
  const dist = posWorld.distance(cameraPosition);

  // Transition parameters (Near/Far thresholds)
  const near = float(10.0);
  const far = float(500.0);
  const factor = smoothstep(far, near, dist);

  return factor;
});

// Perlin Noise Helper Functions in TSL
const mod289 = Fn(({ x }: { x: any }) =>
  x.sub(x.div(289.0).floor().mul(289.0)),
);
const permute = Fn(({ x }: { x: any }) =>
  mod289({ x: x.mul(34.0).add(1.0).mul(x) }),
);
const taylorInvSqrt = Fn(({ r }: { r: any }) =>
  float(1.79284291400159).sub(r.mul(0.85373472095314)),
);
const fade = Fn(({ t }: { t: any }) =>
  t
    .mul(t)
    .mul(t)
    .mul(t.mul(t.mul(6.0).sub(15.0)).add(10.0)),
);

const noise2D = Fn(([p]: [any]) => {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));
  
  const hash = (p: any) => {
    const q = vec2(
      dot(p, vec2(127.1, 311.7)),
      dot(p, vec2(269.5, 183.3))
    );
    return fract(sin(q).mul(43758.5453)).x;
  };
  
  const a = hash(i);
  const b = hash(i.add(vec2(1.0, 0.0)));
  const c = hash(i.add(vec2(0.0, 1.0)));
  const d = hash(i.add(vec2(1.0, 1.0)));
  
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
});

const gerstnerWaveAnalytical = (wave: any, p: any, time: any) => {
  const steepness = wave.z;
  const wavelength = wave.w;
  const k = mul(2.0, 3.14159).div(wavelength);
  const c = sqrt(div(9.81, k));
  const d = normalizeNode(wave.xy);
  const f = k.mul(dot(d, p.xz).sub(c.mul(time)));
  const a = steepness.div(k);

  const cosf = cos(f);
  const sinf = sin(f);

  const x = d.x.mul(a.mul(cosf));
  const y = a.mul(sinf);
  const z = d.y.mul(a.mul(cosf));

  // The analytical derivative of x, y, z wrt spatial position x, z
  const WA = steepness.mul(sinf).negate();
  const WH = steepness.mul(cosf);

  return {
    offset: vec3(x, y, z),
    dDxdx: WA.mul(d.x).mul(d.x),
    dDxdz: WA.mul(d.x).mul(d.y),
    dDzdx: WA.mul(d.y).mul(d.x),
    dDzdz: WA.mul(d.y).mul(d.y),
    dHdx: WH.mul(d.x),
    dHdz: WH.mul(d.y),
  };
};

// Custom Fresnel Node
const fresnelNode = Fn(([power]: [any]) => {
  const viewDir = positionViewDirection;
  const worldNormal = normalWorld;
  return float(1.0).sub(dot(viewDir, worldNormal)).clamp(0, 1).pow(power);
});

export interface WaterMaterialParams {
  envMap: THREE.CubeTexture | null;
  normalMap1: THREE.Texture;
  normalMap2: THREE.Texture;
  foamMap: THREE.Texture;
  vDisp1?: any;
  hDisp1?: any;
  vDisp2?: any;
  hDisp2?: any;
  vDisp3?: any;
  hDisp3?: any;
  j1?: any;
  j2?: any;
  vSlopeX?: any;
  vSlopeZ?: any;
  gridSize?: number;
  simulationL?: number; // The simulation domain size in metres (must match baseL in script.ts)
  /** Hi-Z mip-chain StorageTexture produced by HiZDepthPass.  Required for SSR. */
  hiZTexture?: THREE.Texture;
  /** Total number of mip levels in the Hi-Z chain. */
  hiZMipCount?: number;
}

export const WaterNodeMaterial = (params: WaterMaterialParams) => {
  // ── Gerstner Wave uniforms ─────────────────────────────────────────────────
  const uGerstnerWaveA = uniform(vec4(1.0, 0.2, 0.12, 3.5)); // Fits ~3 times in 10m
  const uGerstnerWaveB = uniform(vec4(0.3, 1.0, 0.1, 5.0)); // Fits 2 times
  const uGerstnerWaveC = uniform(vec4(-0.8, -0.3, 0.08, 7.5));
  const uGerstnerWaveD = uniform(vec4(0.4, -0.7, 0.06, 10.0));

  // ── Beer-Lambert Volumetric Absorption uniforms ───────────────────────────
  // Spectral absorption coefficients (RGB) for each optical constituent.
  // Units are m⁻¹ (attenuation per metre of depth).
  //
  // Pure water:      high red absorption, low blue  (deepest oceans are blue)
  // Phytoplankton:   absorbs blue + red, transmits green (adds bio-green tint)
  //   – scaled by chlorophyll concentration (mg m⁻³)
  // CDOM:            absorbs strongly at short λ (yellow-brown coastal tint)
  //   – scaled by aCDOM reference value

  const uAbsWater = uniform(vec3(0.45, 0.04, 0.01)); // Pure-water α: Absorbs Red, transmits Blue
  const uAbsPhyto = uniform(vec3(0.01, 0.04, 0.02)); // Phyto: Absorbs Green/Blue for photosynthesis
  const uChl = uniform(0.1); // Reduced default chlorophyll for clearer water
  const uAbsNAP = uniform(vec3(0.02, 0.02, 0.01));
  const uNAP = uniform(0.05);
  const uAbsCDOM = uniform(vec3(0.01, 0.04, 0.15)); // CDOM: Absorbs Blue/UV (Yellow Substance)
  const uaCDOM = uniform(0.01); // Reduced for better visibility in shallow diorama
  const uDepthScale = uniform(1.0);
  const uBackscatter = uniform(vec3(0.01, 0.03, 0.06)); // More luminous deep-ocean fog (RGB)
  // Surface tint blended in at zero depth
  const uSurfaceColor = uniform(color("#051a33")); // Slightly brighter navy base

  const uTime = uniform(0.0);

  const uNormalScale = uniform(1.0);
  const uNormalIntensity = uniform(0.2);
  const uUseTextures = uniform(true); // Toggle textures dynamically

  // -------------------------------------------------------------------------
  // Varyings & Inter-Stage Data
  // -------------------------------------------------------------------------
  // These will be initialized in the vertex stage below
  let jacobianVarying: any;
  let normalVarianceVarying: any;
  let waveHeightVarying: any;
  let undisplacedUVVarying: any;

  // -------------------------------------------------------------------------
  // 1. IFFT Wave Sampling (from JONSWAP compute cascades)
  // -------------------------------------------------------------------------
  // The simulation L must match the domain used in JonswapSpectrum (baseL in script.ts).
  // Using positionWorld.xz directly maps world coords into the simulation grid.
  const gridSize = float(params.gridSize || 256);
  const L_sim = float(params.simulationL || 25.0); // simulation domain in metres
  const undisplacedUV = positionWorld.xz.div(L_sim).add(0.5).fract();
  undisplacedUVVarying = varying(undisplacedUV);

  const uvCoord = undisplacedUV;

  /**
   * Bilinear sampling for Storage Buffers
   * Optimized version
   */
  const bilinearSample = Fn(([buffer, uv]: [any, any]) => {
    const scaledUV = uv.mul(gridSize);
    const i00 = scaledUV.sub(0.5).floor();
    const f = scaledUV.sub(0.5).fract();

    const gx0 = i00.x.add(gridSize).mod(gridSize);
    const gx1 = i00.x.add(1).add(gridSize).mod(gridSize);
    const gy0 = i00.y.add(gridSize).mod(gridSize);
    const gy1 = i00.y.add(1).add(gridSize).mod(gridSize);

    const v00 = buffer.element(int(gy0.mul(gridSize).add(gx0)));
    const v10 = buffer.element(int(gy0.mul(gridSize).add(gx1)));
    const v01 = buffer.element(int(gy1.mul(gridSize).add(gx0)));
    const v11 = buffer.element(int(gy1.mul(gridSize).add(gx1)));

    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
  });

  // Vertical and Horizontal Displacements. Scale up to be visible at diorama scale.
  const ifftScale = float(params.simulationL ? params.simulationL / 12.0 : 2.0);
  const v1Sample = params.vDisp1 ? bilinearSample(params.vDisp1, uvCoord) : vec4(0);
  const h1Sample = params.hDisp1 ? bilinearSample(params.hDisp1, uvCoord) : vec4(0);

  const vIFFT = v1Sample.x.mul(ifftScale);
  const hIFFT = vec2(h1Sample.x, h1Sample.z).mul(ifftScale);

  const ifftOffset = vec3(hIFFT.x, vIFFT, hIFFT.y);

  // Jacobian & Slopes (for foam and normals)
  const j1 = params.j1 ? bilinearSample(params.j1, uvCoord).xz : vec2(0);
  const j2 = params.j2 ? bilinearSample(params.j2, uvCoord).xz : vec2(0);

  // Choppiness: scales horizontal displacement (lambda in Tessendorf paper)
  const uChoppiness = uniform(1.2);

  const dDxdx_ifft = j1.x.mul(uChoppiness);
  const dDxdz_ifft = j1.y.mul(uChoppiness);
  const dDzdx_ifft = j2.x.mul(uChoppiness);
  const dDzdz_ifft = j2.y.mul(uChoppiness);

  const vSlopeSample = params.vSlopeX
    ? bilinearSample(params.vSlopeX, uvCoord)
    : vec4(0);
  const dHdx_ifft = vSlopeSample.x;
  const dHdz_ifft = vSlopeSample.z;

  // -------------------------------------------------------------------------
  // Shading & Maritime Aesthetics
  // -------------------------------------------------------------------------

  // SSS uniforms (tunable via debug UI)
  const uSSSColor = uniform(color("#00d4ff")); // Cyan crest glow
  const uSSSStrength = uniform(0.8); // More subtle scattering
  const uSSSPower = uniform(8.0); // Sharper scattering lobe
  const uSSSDistortion = uniform(0.3); // Diffusion factor
  const uSSSThicknessScale = uniform(6.0); // Faster attenuation in volume
  const uFoamThreshold = uniform(0.4);
  const uFoamColor = uniform(color("#e8f4f8"));

  // Sun direction (world space, non-normalized — GUI controls azimuth/elevation)
  const uSunDirection = uniform(vec3(1.0, 0.8, 1.0));

  // Sun specular disk on the water surface
  const uSunColor = uniform(color("#fff5e0"));
  const uSunSpecularPower = uniform(1500.0); // Sharper glint for sun disk
  const uSunSpecularStrength = uniform(30.0); // More balanced glint
  const uResolution = uniform(vec2(1, 1));

  const uEnableSSS = uniform(1.0);
  const uEnableFoam = uniform(1.0);
  const uEnableWaves = uniform(1.0);

  // ── Refraction & Dispersion uniforms ────────────────────────────────────
  // Base IOR for water (1.34 is the physical value for seawater at 550 nm)
  const uIOR = uniform(1.34);
  // Distortion strength in screen-space pixels (scale with viewport size)
  const uRefrStrength = uniform(0.02);
  // Dispersion: additional IOR spread per channel (nR < nG < nB)
  // Real water dispersion dn ≈ 0.006 across visible spectrum
  const uDispersion = uniform(0.003);

  // Diorama logic: isolate the top face using the local normal.
  const isTop = normalLocal.y.greaterThan(0.5);

  // ── GERSTNER WAVES (Analytical Displacement & Derivatives) ──
  const gA = gerstnerWaveAnalytical(uGerstnerWaveA, positionWorld, uTime);
  const gB = gerstnerWaveAnalytical(uGerstnerWaveB, positionWorld, uTime);
  const gC = gerstnerWaveAnalytical(uGerstnerWaveC, positionWorld, uTime);
  const gD = gerstnerWaveAnalytical(uGerstnerWaveD, positionWorld, uTime);

  // Accumulate Displacement (Analytical Gerstner + IFFT Spectrum)
  const waveOffset = gA.offset
    .add(gB.offset)
    .add(gC.offset)
    .add(gD.offset)
    .add(ifftOffset);
  const finalWaveOffset = uEnableWaves
    .greaterThan(0.5)
    .select(waveOffset, vec3(0));

  // To prevent the mesh from tearing at the seams where the top face meets the side walls,
  // we must displace the side walls to perfectly match the top face's wiggling.
  // We use a vertical height mask so the top edge fully follows the waves, while the
  // bottom of the diorama box remains perfectly anchored and flat.
  // positionLocal.y goes from -1.0 to 1.0 for a standard BoxGeometry.
  const verticalMask = positionLocal.y.add(1.0).div(2.0).clamp(0.0, 1.0);
  const p = positionLocal.add(finalWaveOffset.mul(verticalMask));

  // Accumulate Spatial Derivatives
  const dDxdx = gA.dDxdx
    .add(gB.dDxdx)
    .add(gC.dDxdx)
    .add(gD.dDxdx)
    .add(dDxdx_ifft);
  const dDxdz = gA.dDxdz
    .add(gB.dDxdz)
    .add(gC.dDxdz)
    .add(gD.dDxdz)
    .add(dDxdz_ifft);
  const dDzdx = gA.dDzdx
    .add(gB.dDzdx)
    .add(gC.dDzdx)
    .add(gD.dDzdx)
    .add(dDzdx_ifft);
  const dDzdz = gA.dDzdz
    .add(gB.dDzdz)
    .add(gC.dDzdz)
    .add(gD.dDzdz)
    .add(dDzdz_ifft);
  const dHdx = gA.dHdx.add(gB.dHdx).add(gC.dHdx).add(gD.dHdx).add(dHdx_ifft);
  const dHdz = gA.dHdz.add(gB.dHdz).add(gC.dHdz).add(gD.dHdz).add(dHdz_ifft);

  // Calculate the Jacobian Determinant (Used for pinch-breaking foam)
  const rawDetJ = add(1.0, dDxdx).mul(add(1.0, dDzdz)).sub(dDxdz.mul(dDzdx));
  const detJ = uEnableWaves.greaterThan(0.5).select(rawDetJ, float(1.0));

  // Assign varyings for fragment stage (Vertex Stage Logic)
  jacobianVarying = varying(detJ);
  waveHeightVarying = varying(
    finalWaveOffset.y.mul(0.5).add(0.5).clamp(0.0, 1.0),
  );

  // LEADR Mapping: Calculate normal variance for distance-based roughness
  const lod = getLODFactor(positionWorld);
  const slopeVariance = dHdx
    .pow(2.0)
    .add(dHdz.pow(2.0))
    .mul(float(1.0).sub(lod));
  normalVarianceVarying = varying(slopeVariance);

  // Compute True Geometric Normal for the displaced surface
  const Tx = vec3(add(1.0, dDxdx), dHdx, dDzdx);
  const Tz = vec3(dDxdz, dHdz, add(1.0, dDzdz));
  const rawGeoNormal = Tz.cross(Tx).normalize();
  const geoNormal = uEnableWaves
    .greaterThan(0.5)
    .select(rawGeoNormal, vec3(0, 1, 0));

  // Mix the flat side-wall normals with the true top-face displaced normals
  const trueNormalWorld = mix(normalWorld, geoNormal, isTop);

  // ── 1. High-Fidelity Normal Mapping ───────────────────────────────────────
  const surfaceUV = positionWorld.xz;

  // Layer 1: Medium broad ripples
  const nUV1 = surfaceUV
    .mul(uNormalScale.mul(0.1))
    .add(vec2(uTime.mul(0.03), uTime.mul(0.04)));
  const n1 = texture(params.normalMap1, nUV1).xyz.mul(2.0).sub(1.0);

  // Layer 2: Fine detail ripples.
  const s = sin(float(0.8));
  const c = cos(float(0.8));
  const rotUV = vec2(
    surfaceUV.x.mul(c).sub(surfaceUV.y.mul(s)),
    surfaceUV.x.mul(s).add(surfaceUV.y.mul(c)),
  );
  const nUV2 = rotUV
    .mul(uNormalScale.mul(0.1618))
    .sub(vec2(uTime.mul(0.02), uTime.mul(0.01)));
  const n2 = texture(params.normalMap2, nUV2).xyz.mul(2.0).sub(1.0);

  // Blend normals
  const nTangent = normalizeNode(n1.add(n2));

  // Convert Tangent Space (x=Right, y=Up, z=Forward) to World Space (x=Right, y=Up_World, z=Forward_World)
  // Because the water is an XZ plane, Tangent Z (out of texture) maps to World Y (up). Tangent Y maps to World Z.
  const nWorld = vec3(nTangent.x, nTangent.z, nTangent.y).mul(isTop);

  // Shading normal uses the true physically calculated geometry normal!
  // If textures are disabled, we gracefully fall back to the pure mathematical surface normal.
  const shadingNormal = uUseTextures.select(
    trueNormalWorld.add(nWorld.mul(uNormalIntensity)).normalize(),
    trueNormalWorld,
  );

  /* --- COMMENTED OUT ADVANCED FEATURES BY DEFAULT ---
// // ── 2. High-Fidelity Foam Integration ─────────────────────────────────────
//   const foamUV = surfaceUV
//     .mul(0.1)
//     .add(nTangent.xy.mul(0.02))
//     .sub(vec2(uTime.mul(0.01)));
// 
//   // If textures are disabled, use a neutral mathematical constant (0.5)
//   // so the foam forms smooth geometric edges at the breaking crests.
//   const foamTex = uUseTextures.select(
//     texture(params.foamMap as THREE.Texture, foamUV).r,
//     float(0.5),
//   );
// 
//   // Mask foam using the Jacobian determinant (passed via varying).
//   // jacobianVarying drops below 1.0 at sharp crests, representing wave 'pinching' or breaking.
//   const foamTrigger = float(1.0).sub(jacobianVarying);
//   const organicTrigger = foamTrigger.add(foamTex.mul(0.2));
// 
//   // FIXED: Properly scale smoothstep bounds so foam ONLY appears at crests.
//   const foamHeightMask = smoothstep(
//     uFoamThreshold,
//     uFoamThreshold.add(0.2),
//     organicTrigger,
//   );
// 
//   // Final Foam: Masked to the top face ONLY so it doesn't bleed down the side walls!
//   const finalFoamMask = uEnableFoam
//     .greaterThan(0.5)
//     .select(foamHeightMask.mul(foamTex).mul(isTop), float(0));
//   const foamColor = uFoamColor.mul(finalFoamMask).mul(0.9);
// 
//   // ── Vectors & Two-Sided Shading ───────────────────────────────────────────
//   // V: fragment → camera (world space)
//   const V = normalizeNode(cameraPosition.sub(positionWorld));
//   const L = normalizeNode(uSunDirection);
// 
//   // Determine if we are viewing the surface from above or below (underwater).
//   // CRITICAL: This 'underwater' logic must ONLY apply to the top water surface.
//   // The side-walls of the diorama box should always be treated as 'outside' looking in.
//   const vDotN = dot(V, shadingNormal);
//   const isBelow = vDotN.lessThan(0.0).and(isTop);
// 
//   // Flip the normal for the underside so lighting and Fresnel work correctly
//   const faceNormal = isBelow.select(shadingNormal.negate(), shadingNormal);
// 
//   // Helper: Convert non-linear depth to view-space distance (linear)
//   const getLinearDepth = Fn(([depthSample]: [any]) => {
//     const viewZ = cameraFar
//       .mul(cameraNear)
//       .div(cameraFar.sub(cameraNear).mul(depthSample).sub(cameraFar));
// 
//     // WebGPU uses Reverse-Z (depth=0 is the far plane). Standard WebGL uses depth=1.
//     // If the ray hits the skybox, we must force the depth to a massive number so the water
//     // correctly becomes an opaque deep-ocean cutaway, rather than becoming 0m thick and perfectly clear!
//     const isSkybox = step(depthSample, float(0.001)).add(
//       step(float(0.999), depthSample),
//     );
//     return isSkybox.greaterThan(0.0).select(cameraFar, viewZ.negate());
//   });
// 
//   // ── Two-Ray Beer-Lambert Volumetric Absorption ──────────────────────────
//   // 1. Calculate actual vertical depth using the depth buffer
//   const sceneDepthNonLinear = viewportDepthTexture(viewportUV).r;
//   const sceneDepthLinear = getLinearDepth(sceneDepthNonLinear);
// 
//   // positionView.z is the water surface depth in view space
//   const surfaceDepthLinear = positionView.z.negate();
// 
//   // 2. Account for path length (Two-Ray Model)
//   // Path = Depth / cos(theta). We use the vertical (0,1,0) for the surface plane.
//   const worldUp = vec3(0, 1, 0);
//   const cosView = dot(V, worldUp).abs().clamp(0.01, 1.0);
//   const cosLight = dot(L, worldUp).abs().clamp(0.01, 1.0);
// 
//   // Beer-Lambert vertical depth needs to be handled differently when underwater!
//   // If above: depth is distance between surface and scene.
//   // If below: depth is distance between eye and surface (ignored for scene refraction).
//   const verticalDepth = isBelow.select(
//     float(0.0),
//     sceneDepthLinear.sub(surfaceDepthLinear).max(0.0)
//   ).mul(uDepthScale);
// 
//   const pathView = isBelow.select(
//     positionView.z.negate(), // Distance from eye to surface
//     verticalDepth.div(cosView)
//   );
// 
//   const pathLight = isBelow.select(
//     float(0.0), // Light enters from above, path in water is already covered by pathView
//     verticalDepth.div(cosLight)
//   );
// 
//   const totalPath = pathView.add(pathLight);
// 
//   // Total spectral absorption: α_total(λ) = αw + αp·Chl + α_NAP·NAP + α_CDOM·aCDOM
//   const alphaTotalR = uAbsWater.x
//     .add(uAbsPhyto.x.mul(uChl))
//     .add(uAbsNAP.x.mul(uNAP))
//     .add(uAbsCDOM.x.mul(uaCDOM));
//   const alphaTotalG = uAbsWater.y
//     .add(uAbsPhyto.y.mul(uChl))
//     .add(uAbsNAP.y.mul(uNAP))
//     .add(uAbsCDOM.y.mul(uaCDOM));
//   const alphaTotalB = uAbsWater.z
//     .add(uAbsPhyto.z.mul(uChl))
//     .add(uAbsNAP.z.mul(uNAP))
//     .add(uAbsCDOM.z.mul(uaCDOM));
//   const alphaTotal = vec3(alphaTotalR, alphaTotalG, alphaTotalB);
// 
//   // Beer-Lambert transmittance: T(λ) = exp(-α(λ) · totalPath)
//   const transmittance = exp(alphaTotal.negate().mul(totalPath));
// 
//   // Backscatter: luminous curtain integrated along view path
//   const backscatterContrib = uBackscatter.mul(float(1.0).sub(transmittance));
// 
//   // ── Frostbite 2 Subsurface Scattering ────────────────────────────────────
//   // Barré-Brisebois/Bouchard: I_sss = max(0, V · (-L + N_dist))^p
//   // We distort the light vector with the normal to simulate internal diffusion
//   const distortedL = normalizeNode(
//     L.negate().add(shadingNormal.mul(uSSSDistortion)),
//   );
//   const directionalSSS = dot(V, distortedL).clamp(0.0, 1.0).pow(uSSSPower);
// 
//   // Add an ambient base glow so the SSS is always visible on the crests
//   // regardless of where the camera is positioned relative to the sun.
//   const scatterLobe = directionalSSS.add(0.3).clamp(0.0, 1.0);
// 
//   // Thickness proxy: waves are thinner at crests and thicker in troughs.
//   // We use the interpolated wave height from the vertex stage (waveHeightVarying).
//   const waveHeightNormalized = waveHeightVarying;
// 
//   // We use the exponential model: exp(-T * (1 - H))
//   const sssAttenuation = exp(
//     uSSSThicknessScale.mul(waveHeightNormalized.sub(1.0)),
//   );
//   const rawSSS = uSSSColor
//     .mul(scatterLobe)
//     .mul(sssAttenuation)
//     .mul(uSSSStrength);
// 
//   const sssContrib = uEnableSSS
//     .greaterThan(0.5)
//     .select(rawSSS, vec3(0));
// 
//   // ── Screen-Space Refraction + Chromatic Dispersion ──────────────────────
//   // We use Snell's Law (via refract()) to find the actual light-bending vector.
//   // Above (Air to Water): 1.0 / IOR
//   // Below (Water to Air): IOR / 1.0
//   const eta = isBelow.select(uIOR, float(1.0).div(uIOR));
// 
//   // ── Fresnel & Environment Reflection ──────────────────────────────────────
//   // Fresnel F0 calculation based on uIOR: ((1-n)/(1+n))^2
//   const f0 = uIOR.sub(1.0).div(uIOR.add(1.0)).pow(2.0);
// 
//   // Physically accurate Schlick's approximation using the face-aligned normal
//   const oneMinusCos = float(1.0).sub(dot(V, faceNormal)).clamp(0.0, 1.0);
//   const fRaw = f0.add(float(1.0).sub(f0).mul(oneMinusCos.pow(5.0)));
// 
//   // Handle Total Internal Reflection (TIR) physically:
//   // If the refraction vector is zero, Fresnel reflection MUST be 100%.
//   const R_ior_fresnel = refract(V.negate(), faceNormal, eta);
//   const isTIR = dot(R_ior_fresnel, R_ior_fresnel).lessThanEqual(0.0);
// 
//   const f = isTIR.select(float(1.0), fRaw);
// 
//   const reflectDir = reflect(V.negate(), faceNormal);
//   
//   const envReflection = cubeTexture(
//     params.envMap as THREE.CubeTexture,
//     reflectDir,
//   );
// 
//   // When viewing from below (underwater), TIR reflects the deep ocean, NOT the skybox!
//   // So we use the pure water surface tint instead of the env map.
//   const reflectionColor = isBelow.select(
//     uSurfaceColor,
//     envReflection
//   );
// 
//   const getRefractedUV = Fn(([iorRatio]: [any]) => {
//     // V: View vector (from fragment to camera)
//     // faceNormal: Normal pointing towards the viewer
//     const R = refract(V.negate(), faceNormal, iorRatio);
// 
//     // If TIR (Total Internal Reflection) occurs, refract returns zero.
//     // In screen-space refraction, we just skip the distortion offset.
//     const hasRefraction = dot(R, R).greaterThan(0.0);
// 
//     // Project the refracted world-space vector into screen-space offset
//     // We use normalView.xy as a shortcut for screen-space distortion direction
//     const distortion = normalView.xy.mul(uRefrStrength);
//     return hasRefraction.select(
//       viewportUV.add(distortion.mul(R.xz.length())),
//       viewportUV
//     );
//   });
// 
//   const uvR = getRefractedUV(eta.sub(uDispersion));
//   const uvG = getRefractedUV(eta);
//   const uvB = getRefractedUV(eta.add(uDispersion));
// 
//   const sceneR = viewportSharedTexture(uvR).r;
//   const sceneG = viewportSharedTexture(uvG).g;
//   const sceneB = viewportSharedTexture(uvB).b;
//   const refractedScene = vec3(sceneR, sceneG, sceneB);
// 
//   // Apply Beer-Lambert tinting to the refracted underwater view.
//   // Instead of multiplying to black, we mix the scene with the deep-water surface color!
//   // Crests (thin water, T≈1) are clear; deep troughs absorb heavily into the surface tint.
//   const refractedColor = mix(uSurfaceColor, refractedScene, transmittance).add(
//     backscatterContrib,
//   );
// 
//   // ── Sun Specular Disk (Blinn-Phong) ──────────────────────────────────────
//   // The sun appears as a bright sharp glint on the rippled water surface.
//   // Using the half-vector H between view and light for a tight specular lobe.
//   const H = normalizeNode(V.add(L));
//   const NdotH = dot(shadingNormal, H).clamp(0.0, 1.0);
//   // pow() with high exponent makes this near-zero everywhere except the actual glint pixel.
//   // Clamp the final contribution so it never produces a global color tint.
//   const rawSunSpec = pow(NdotH, uSunSpecularPower)
//     .mul(uSunSpecularStrength)
//     .clamp(0.0, 1.0);
//   const sunSpecular = uSunColor.mul(rawSunSpec);
// 
//   // ── Final Composite ────────────────────────────────────────────────────────
//   // Below surface: refracted (absorbed) underwater view + SSS crest glow
//   // Above (grazing): Fresnel reflection to env-map
//   // Sun specular disk and foam are always on top.
//   const underWater = refractedColor.add(sssContrib);
//   const litColor = mix(underWater, reflectionColor, f);
// 
//   // Directional Atmospheric Fog
//   const fogDensity = float(0.02);
//   const dist = cameraPosition.sub(positionWorld).length();
//   const viewDotUp = dot(V, vec3(0, 1, 0)).abs();
//   // `(1.0 - viewDotUp)^4` approximates `(1.0 - cos(theta))^4` where theta is angle to horizon.
//   const fogFactor = float(1.0)
//     .sub(exp(dist.pow(2.0).negate().mul(fogDensity.pow(2.0))))
//     .mul(float(1.0).sub(viewDotUp).pow(4.0))
//     .mul(isBelow.select(float(0.0), float(1.0)));
//     
//   // Sample sky at horizon (using the view direction, which is V.negate())
//   const viewDir = V.negate();
//   const horizonVec = vec3(viewDir.x, 0.001, viewDir.z).normalize();
//   const skyHorizonColor = cubeTexture(params.envMap as THREE.CubeTexture, horizonVec);
// 
//   let finalColor = litColor
//     .add(foamColor)
//     .add(sunSpecular);
//     
//   finalColor = mix(finalColor, skyHorizonColor, fogFactor);
--- END COMMENTED OUT FEATURES --- */

  // ── Vectors ──────────────────────────────────────────────────────────────
  const V = normalizeNode(cameraPosition.sub(positionWorld));
  const L = normalizeNode(uSunDirection);

  // ── Wave Height Function (for Volumetrics) ────────────────────────────────
  const getWaveHeightAt = Fn(([xz]: [any]) => {
    // For ray-marching steps, we approximate the height.
    // However, for the very start of the ray, we want it to match perfectly.
    const uv = xz.div(L_sim).add(0.5).fract();
    const v = params.vDisp1 ? bilinearSample(params.vDisp1, uv).x.mul(ifftScale) : float(0);

    const kA = mul(2.0, 3.14159).div(uGerstnerWaveA.w);
    const cA = sqrt(div(9.81, kA));
    const dA = normalizeNode(uGerstnerWaveA.xy);
    const fA = kA.mul(dot(dA, xz).sub(cA.mul(uTime)));
    const yA = uGerstnerWaveA.z.div(kA).mul(sin(fA));

    const kB = mul(2.0, 3.14159).div(uGerstnerWaveB.w);
    const cB = sqrt(div(9.81, kB));
    const dB = normalizeNode(uGerstnerWaveB.xy);
    const fB = kB.mul(dot(dB, xz).sub(cB.mul(uTime)));
    const yB = uGerstnerWaveB.z.div(kB).mul(sin(fB));

    const kC = mul(2.0, 3.14159).div(uGerstnerWaveC.w);
    const cC = sqrt(div(9.81, kC));
    const dC = normalizeNode(uGerstnerWaveC.xy);
    const fC = kC.mul(dot(dC, xz).sub(cC.mul(uTime)));
    const yC = uGerstnerWaveC.z.div(kC).mul(sin(fC));

    const kD = mul(2.0, 3.14159).div(uGerstnerWaveD.w);
    const cD = sqrt(div(9.81, kD));
    const dD = normalizeNode(uGerstnerWaveD.xy);
    const fD = kD.mul(dot(dD, xz).sub(cD.mul(uTime)));
    const yD = uGerstnerWaveD.z.div(kD).mul(sin(fD));

    return yA.add(yB).add(yC).add(yD).add(v).add(1.0);
  });

  const uFogDensity = uniform(1.5);
  const uVolumeAO = uniform(1.2);
  const uEnableVolumetricFog = uniform(1.0);

  // ── Skybox (Reflection) ──────────────────────────────────────────────────
  const reflectDir = reflect(V.negate(), shadingNormal);
  const envReflection = cubeTexture(
    params.envMap as THREE.CubeTexture,
    reflectDir,
  );

  // ── Fresnel Reflection ───────────────────────────────────────────────────
  // fresnel = 0.02 + 0.98 * (1.0 - theta_i)^5
  const theta_i = dot(V, shadingNormal).abs().clamp(0.0, 1.0);
  const fresnel = float(0.02).add(float(0.98).mul(float(1.0).sub(theta_i).pow(5.0)));

  // ── Screen-Space Refraction + Chromatic Dispersion ───────────────────────
  // IOR ratio for air-to-water transition (above surface only).
  // eta = 1/n  (light going from air into water)
  const eta = float(1.0).div(uIOR);

  // Compute a refracted screen-UV for a given per-channel IOR ratio.
  // Strategy: use Snell's law (refract()) to get the world-space bending vector,
  // then project its XZ deviation into screen-space via the tangent-space normal.
  // normalView.xy is the view-space tangent deviation — a reliable proxy for
  // screen-space distortion direction without requiring a full VP matrix multiply.
  const getRefractedUV = Fn(([iorRatio]: [any]) => {
    const R = refract(V.negate(), shadingNormal, iorRatio);
    // If TIR (Total Internal Reflection) — refract() returns zero vector.
    // Fall back to the un-distorted UV so we never sample garbage.
    const hasRefraction = dot(R, R).greaterThan(0.0);
    // Scale by the horizontal magnitude of the refracted ray to get stronger
    // distortion at grazing angles and near-zero distortion at normal incidence.
    const distortion = normalView.xy.mul(uRefrStrength).mul(R.xz.length());
    return hasRefraction.select(viewportUV.add(distortion), viewportUV);
  });

  // Sample the background three times — one per colour channel — at slightly
  // different IOR ratios to simulate wavelength-dependent dispersion.
  // Red bends least (IOR - dispersion), Blue bends most (IOR + dispersion).
  const uvR = getRefractedUV(eta.sub(uDispersion));
  const uvG = getRefractedUV(eta);
  const uvB = getRefractedUV(eta.add(uDispersion));

  const sceneR = viewportSharedTexture(uvR).r;
  const sceneG = viewportSharedTexture(uvG).g;
  const sceneB = viewportSharedTexture(uvB).b;
  const background = vec3(sceneR, sceneG, sceneB);

  // ── Volumetric Absorption (Beer-Lambert) ─────────────────────────────────

  // Helper: Convert non-linear depth to view-space distance (linear)
  const getLinearDepth = Fn(([depthSample]: [any]) => {
    const viewZ = cameraFar
      .mul(cameraNear)
      .div(cameraFar.sub(cameraNear).mul(depthSample).sub(cameraFar));

    // WebGPU uses Reverse-Z (depth=0 is the far plane).
    // If the ray hits the skybox, force depth to a massive number.
    const isSkybox = step(depthSample, float(0.001)).add(step(float(0.999), depthSample));
    return isSkybox.greaterThan(0.0).select(cameraFar, viewZ.negate());
  });
  // Total spectral absorption: α_total(λ) = αw + αp·Chl + α_NAP·NAP + α_CDOM·aCDOM
  const alphaTotal = uAbsWater
    .add(uAbsPhyto.mul(uChl))
    .add(uAbsNAP.mul(uNAP))
    .add(uAbsCDOM.mul(uaCDOM));

  // 1. Calculate actual vertical depth using the depth buffer
  const sceneDepthNonLinear = viewportDepthTexture(viewportUV).r;
  const sceneDepthLinear = getLinearDepth(sceneDepthNonLinear);
  const surfaceDepthLinear = positionView.z.negate();

  // 2. Account for path length
  const worldUp = vec3(0, 1, 0);
  const cosView = dot(V, worldUp).abs().clamp(0.01, 1.0);
  const verticalDepth = sceneDepthLinear.sub(surfaceDepthLinear).max(0.0).mul(uDepthScale);
  
  // Calculate end position for the ray march (approximate point on the floor)
  // We ray-march along the refraction vector for physical accuracy
  const R = refract(V.negate(), shadingNormal, eta);
  const hasRefraction = dot(R, R).greaterThan(0.0);
  const rayDir = hasRefraction.select(R, V.negate());
  
  // Dist to floor along rayDir: d = verticalDepth / cos(theta)
  // where theta is angle to vertical. cos(theta) = rayDir.y.abs()
  const pathLength = verticalDepth.div(rayDir.y.abs().max(0.01));
  const endPos = positionWorld.add(rayDir.mul(pathLength));

  // ── Ray-Marched Volumetric Absorption ────────────────────────────────────
  const volumetricResult = Fn(() => {
    // Optimization: Reduce step count from 12 to 8.
    // We use a dithered offset to hide banding, which is much cheaper than more steps.
    const stepCount = int(8);
    const accumColor = vec3(0).toVar();
    const currentTransmittance = vec3(1.0).toVar();

    const marchRay = endPos.sub(positionWorld);
    const totalDist = marchRay.length();
    const stepSize = totalDist.div(float(stepCount));
    const stepDir = marchRay.normalize().mul(stepSize);

    // Dither the start position to break up banding artifacts.
    // Standard 2x2 Bayer-like pattern using screen coordinates.
    const dither = fract(dot(viewportUV, uResolution.mul(0.75))).mul(stepSize);
    const p = positionWorld.add(marchRay.normalize().mul(dither)).toVar();

    Loop({ start: int(0), end: stepCount }, () => {
      const h = getWaveHeightAt(p.xz);
      const isBelow = p.y.lessThan(h.add(0.01)); 

      const depthAtP = h.sub(p.y).max(0.0);
      const ao = exp(depthAtP.mul(uVolumeAO.negate()));

      const alpha = isBelow.select(alphaTotal.mul(uDepthScale), vec3(0));
      const stepTransmittance = exp(alpha.negate().mul(stepSize));
      const stepFog = isBelow.select(uBackscatter.mul(float(1.0).sub(stepTransmittance)).mul(ao).mul(uFogDensity), vec3(0));

      accumColor.addAssign(stepFog.mul(currentTransmittance));
      currentTransmittance.mulAssign(stepTransmittance);
      p.addAssign(stepDir);
    });

    const marchColor = mix(uSurfaceColor.xyz, background, currentTransmittance).add(accumColor);
    const marchAvgT = currentTransmittance.r.add(currentTransmittance.g).add(currentTransmittance.b).div(3.0);

    // Fallback to simple Beer-Lambert if volumetric is disabled
    const simpleAlpha = alphaTotal.mul(uDepthScale);
    const simpleT = exp(simpleAlpha.negate().mul(pathLength));
    const simpleFog = uBackscatter.mul(float(1.0).sub(simpleT));
    const simpleColor = mix(uSurfaceColor.xyz, background, simpleT).add(simpleFog);
    const simpleAvgT = simpleT.r.add(simpleT.g).add(simpleT.b).div(3.0);

    return uEnableVolumetricFog.greaterThan(0.5).select(
        vec4(marchColor, marchAvgT),
        vec4(simpleColor, simpleAvgT)
    );
  })();


  const oceanColor = volumetricResult.rgb;
  const avgTransmittance = volumetricResult.a;

  // ── Frostbite 2 Subsurface Scattering ────────────────────────────────────
  // Barré-Brisebois/Bouchard: I_sss = max(0, V · (-L + N_dist))^p
  // We distort the light vector with the normal to simulate internal diffusion
  const distortedL = normalizeNode(
    L.negate().add(shadingNormal.mul(uSSSDistortion)),
  );
  const directionalSSS = dot(V, distortedL).clamp(0.0, 1.0).pow(uSSSPower);

  // Add an ambient base glow so the SSS is always visible on the crests
  // regardless of where the camera is positioned relative to the sun.
  const scatterLobe = directionalSSS.add(0.3).clamp(0.0, 1.0);

  // Thickness proxy: waves are thinner at crests and thicker in troughs.
  // We use the interpolated wave height from the vertex stage (waveHeightVarying).
  const waveHeightNormalized = waveHeightVarying;

  // We use the exponential model: exp(-T * (1 - H))
  const sssAttenuation = exp(
    uSSSThicknessScale.mul(waveHeightNormalized.sub(1.0)),
  );
  const rawSSS = uSSSColor.xyz
    .mul(scatterLobe)
    .mul(sssAttenuation)
    .mul(uSSSStrength);

  const sssContrib = uEnableSSS
    .greaterThan(0.5)
    .select(rawSSS, vec3(0));

  // ── Analytic Caustics ─────────────────────────────────────────────────────
  //
  // Physical basis: the Jacobian determinant det(J) of the wave displacement
  // field measures how much the surface is focusing or diverging refracted
  // sunlight.  Where det(J) < 1 the surface is converging rays → bright
  // caustic spot.  Where det(J) > 1 it's diverging → dark band.
  //
  // Caustic intensity = max(0,  1/det(J) - 1 )  (excess above uniform)
  // clamped and scaled, then attenuated by:
  //   • Sun-to-normal alignment (LdotUp) — only sunlit fragments
  //   • Beer-Lambert transmittance — caustics fade in deep/turbid water
  //   • isTop mask — only the top water face casts caustics downward
  //
  // The result is blended additively into `oceanColor` (the refracted
  // background) so it brightens the sea-floor seen through the water.

  // Uniforms
  const uCausticsStrength  = uniform(0.5);   // peak brightness multiplier
  const uCausticsSharpness = uniform(4.0);   // power — higher = tighter spots
  const uCausticsColor     = uniform(vec3(0.9, 0.97, 1.0)); // slight blue-white tint
  const uEnableCaustics    = uniform(1.0);

  // det(J) from the vertex stage varying (same one used for foam).
  // Values < 1 → focusing; clamp to avoid division by zero at breaking crests.
  const detJ_safe = jacobianVarying.max(float(0.05));

  // Raw focusing factor: (1/detJ)^sharpness gives sharp bright spots.
  // We clamp the result to prevent "white cloud" saturation artifacts in deep water.
  const focusing = float(1.0)
    .div(detJ_safe)
    .pow(uCausticsSharpness)
    .sub(1.0)
    .clamp(0.0, 50.0);

  // Attenuate by how directly the sun hits the surface from above
  const LdotUp = dot(L, vec3(0.0, 1.0, 0.0)).clamp(0.0, 1.0);

  // Depth attenuation — reuse Beer-Lambert transmittance (already computed).
  // We use the square of the average transmittance to make caustics fade 
  // faster than the general water tint, preventing "milky" volume artifacts.
  const causticsDepthAtten = avgTransmittance.pow(2.0);

  // Skybox Mask: Only apply caustics if we hit a floor/object (depth < 0.999).
  // This prevents caustics from appearing as white patterns on the infinite horizon.
  const isBackground = step(float(0.999), sceneDepthNonLinear);
  const floorMask = float(1.0).sub(isBackground);

  const rawCaustics = uCausticsColor
    .mul(focusing)
    .mul(LdotUp)
    .mul(causticsDepthAtten)
    .mul(uCausticsStrength)
    .mul(isTop)
    .mul(floorMask);

  const causticsContrib = uEnableCaustics
    .greaterThan(0.5)
    .select(rawCaustics, vec3(0));

  // Add caustics to the refracted background before the Fresnel composite
  // so they appear on the sea floor / objects seen through the water.
  const oceanColorWithCaustics = oceanColor.add(causticsContrib);

  // ── Fresnel + cubemap reflection composite ───────────────────────────────
  // Schlick Fresnel from the physical IOR: f0 = ((n-1)/(n+1))²
  const f0IOR      = uIOR.sub(1.0).div(uIOR.add(1.0)).pow(2.0);
  const VdotN      = dot(V, shadingNormal).abs().clamp(0.0, 1.0);
  const fresnelSSR = f0IOR.add(float(1.0).sub(f0IOR).mul(float(1.0).sub(VdotN).pow(5.0)));

  // Final Fresnel composite: Beer-Lambert + SSS + caustics below, cubemap above.
  const litColor = mix(oceanColorWithCaustics.add(sssContrib), envReflection.xyz, fresnelSSR);

  // ── Specular (LEADR Mapping + GGX PBR) ───────────────────────────────────
  // Map the loss of high-frequency normal variance into an additive modifier
  // applied to the base PBR microfacet roughness parameter to suppress aliasing
  const H = normalizeNode(V.add(L));
  const NdotH = dot(shadingNormal, H).clamp(0.0, 1.0);
  
  // LEADR Mapping: Incorporate normal variance into roughness
  const baseRoughness = float(0.05);
  const effectiveRoughness = baseRoughness
    .pow(2.0)
    .add(normalVarianceVarying)
    .sqrt()
    .clamp(0, 1);

  // alpha = roughness^2
  const alpha = effectiveRoughness.pow(2.0);
  const alphaSq = alpha.pow(2.0).max(0.00001); // Avoid division by zero
  
  // GGX NDF: D(h) = alpha^2 / (PI * ((N.H)^2 * (alpha^2 - 1) + 1)^2)
  const denom = NdotH.pow(2.0).mul(alphaSq.sub(1.0)).add(1.0);
  const D = alphaSq.div(float(Math.PI).mul(denom.pow(2.0)));
  
  // Energy conservation: The specular lobe is inherently normalized by D.
  const NdotL = dot(shadingNormal, L).clamp(0.0, 1.0);
  const specular = D.mul(NdotL);
  const sunSpecular = uSunColor.xyz.mul(specular).mul(float(1.5)); // Scale intensity for aesthetic balance

  // ── A bit of fog ─────────────────────────────────────────────────────────
  // fog = 1.0 - exp(-depth^2 * fogDensity^2) * (1.0 - cos(theta))^4
  const depth = cameraPosition.sub(positionWorld).length();
  const cosTheta = dot(V, vec3(0, 1, 0)).abs();
  const fogDensity = float(0.02);
  const fogFactor = float(1.0)
    .sub(exp(depth.pow(2.0).negate().mul(fogDensity.pow(2.0))))
    .mul(float(1.0).sub(cosTheta).clamp(0.0, 1.0).pow(4.0))
    .clamp(0.0, 1.0);
  
  const viewDir = V.negate();
  const horizonVec = vec3(viewDir.x, 0.001, viewDir.z).normalize();
  const skyHorizonColor = cubeTexture(params.envMap as THREE.CubeTexture, horizonVec).xyz;

  // ── Foam (Jacobian Mask) ─────────────────────────────────────────────────
  const detJ_frag = jacobianVarying;
  const foamThreshold = uFoamThreshold;
  
  // 1. Dual-Scale Tiling Killer: Use two foam layers at different scales and a 45° rotation offset.
  // This prevents the layers from ever aligning and forming a visible grid.
  const foamUV_A = surfaceUV.mul(0.12).add(vec2(uTime.mul(0.015)));
  const f1 = texture(params.foamMap as THREE.Texture, foamUV_A).r;
  
  const angle = 0.785; // 45 degrees
  const s_rot = Math.sin(angle);
  const c_rot = Math.cos(angle);
  const rotatedUV = vec2(
    surfaceUV.x.mul(c_rot).sub(surfaceUV.y.mul(s_rot)),
    surfaceUV.x.mul(s_rot).add(surfaceUV.y.mul(c_rot))
  );
  const foamUV_B = rotatedUV.mul(0.19).sub(vec2(uTime.mul(0.02)));
  const f2 = texture(params.foamMap as THREE.Texture, foamUV_B).r;
  
  const combinedTex = f1.mul(f2).mul(1.5).add(f1.mul(0.2));

  // 2. Macro-Clumping & Noise Detail: 
  // We use multiple octaves of noise to add organic variety to the foam density
  // and break up the texture pattern.
  const noiseUV = surfaceUV.mul(0.4).add(vec2(uTime.mul(0.02)));
  const fbmNoise = noise2D(noiseUV).mul(0.5)
    .add(noise2D(noiseUV.mul(2.03)).mul(0.25))
    .add(noise2D(noiseUV.mul(4.07)).mul(0.125));

  // Organic Edge Erosion: 
  // Jitter the Jacobian determinant with BOTH the texture and the procedural noise.
  // This makes the foam break up into bubbly filaments and "lacy" structures.
  const organicDetJ = detJ_frag
    .sub(combinedTex.mul(0.15))
    .sub(fbmNoise.mul(0.2));
  
  // 4. Two-Stage Masking for soft transitions
  const denseMask = smoothstep(foamThreshold.add(0.1), foamThreshold, organicDetJ);
  const thinMask = smoothstep(foamThreshold.add(0.4), foamThreshold, organicDetJ).mul(0.3);
  
  // Mix texture and noise: The noise creates the large-scale "clumps" while 
  // the texture provides the high-frequency "bubbles".
  const finalMask = denseMask.max(thinMask).mul(mix(combinedTex, fbmNoise, 0.4));
  
  const foamVisibility = finalMask.mul(uEnableFoam).mul(isTop);

  // 5. Natural Blending: Mix the foam color into the water color rather than adding it.
  let finalColor = mix(litColor, uFoamColor.xyz, foamVisibility).add(sunSpecular);
  finalColor = mix(finalColor, skyHorizonColor, fogFactor);

  // ── Opacity uniform ───────────────────────────────────────────────────────
  const uOpacity = uniform(1.0);

  return {
    positionNode: p,
    colorNode: finalColor,
    transparent: true,
    opacity: uOpacity,
    uniforms: {
      uGerstnerWaveA,
      uGerstnerWaveB,
      uGerstnerWaveC,
      uGerstnerWaveD,
      uSurfaceColor,
      uAbsWater,
      uAbsPhyto,
      uChl,
      uAbsNAP,
      uNAP,
      uAbsCDOM,
      uaCDOM,
      uDepthScale,
      uBackscatter,
      uNormalScale,
      uNormalIntensity,
      uUseTextures,
      uSSSColor,
      uSSSStrength,
      uSSSPower,
      uSSSDistortion,
      uSSSThicknessScale,
      uFoamThreshold,
      uFoamColor,
      uEnableSSS,
      uEnableFoam,
      uEnableWaves,
      uIOR,
      uRefrStrength,
      uDispersion,
      uFogDensity,
      uVolumeAO,
      uEnableVolumetricFog,
      uCausticsStrength,
      uCausticsSharpness,
      uCausticsColor,
      uEnableCaustics,
      uOpacity,
      uTime,
      uSunDirection,
      uChoppiness,
      uSunColor,
      uSunSpecularPower,
      uSunSpecularStrength,
      uResolution,
    },
  };
};
