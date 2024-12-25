
@group(0) @binding(0)
var<uniform> _N : u32;

@group(0) @binding(1)
var<uniform> _Seed : i32;

@group(0) @binding(2)
var<uniform> _LengthScale0 : u32;

@group(0) @binding(3)
var<uniform> _LengthScale1 : u32;

@group(0) @binding(4)
var<uniform> _LengthScale2 : u32;

@group(0) @binding(5)
var<uniform> _LowCutoff : f32;

@group(0) @binding(6)
var<uniform> _HighCutoff : f32;

@group(0) @binding(7)
var<uniform> _Gravity : f32;

@group(0) @binding(8)
var<uniform> _RepeatTime : f32;

@group(0) @binding(9)
var<uniform> _FrameTime : f32;

@group(0) @binding(10)
var<uniform> _Lambda : vec2<f32>;


struct SpectrumParameters {
    scale: f32,
    angle: f32,
    spreadBlend: f32,
    swell: f32,
    alpha: f32,
    peakOmega: f32,
    gamma: f32,
    shortWavesFade: f32,
};

@group(0) @binding(11)
var<storage, read> _Spectrums: array<SpectrumParameters>;

@group(0) @binding(12)
var _SpectrumTextures: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(13)
var _InitialSpectrumTextures: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(14)
var _DisplacementTextures: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(15)
var<uniform> _Depth : f32;

const PI = 3.141492;

//fn hash(value: u32) -> u32 {
//    var state = value;
//    state = state ^ 2747636419u;
//    state = state * 2654435769u;
//    state = state ^ state >> 16u;
//    state = state * 2654435769u;
//    state = state ^ state >> 16u;
//    state = state * 2654435769u;
//    return state;
//}
//
//fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
//    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
//}
//
//fn EulerFormula(x: f32) -> vec2<f32> {
//    return vec2(cos(x), sin(x));
//}
//
//fn UniformToGaussian(u1: f32, u2: f32) -> vec2<f32>{
//    let R = sqrt(-2.0 * log(u2));
//    let theta = 2.0 * PI * u2;
//    return vec2(R * cos(theta), R * sin(theta));
//}
//
//fn Dispersion (kMag: f32) -> f32{
//return sqrt(_Gravity * kMag * tanh(min(kMag * _Depth, 20.0)));
//}
//
//fn DispersionDerivative(kMag: f32) -> f32{
//   let th = tanh(min(kMag * _Depth, 20.0));
//   let ch = cosh(kMag * _Depth);
//   return _Gravity * (_Depth * kMag / ch / ch + th) / Dispersion(kMag) / 2.0;
//}
//
//fn NormalizationFactor(s: f32) -> f32{
//    let s2 = s * s;
//    let s3 = s2 * s;
//    let s4 = s3 * s;
//    if(s<5.0){
//        return -0.00564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163;
//    }
//
//        return -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.90e-02 * s + 3.93e-01;
//}
//
//fn Cosine2s(theta: f32, s: f32) -> f32{
//    return NormalizationFactor(s) * pow(abs(cos(0.5*theta)), 2.0 * s);
//}
//
//fn SpreadPower(omega: f32, peakOmega: f32) -> f32{
//    if(omega > peakOmega){
//        return 9.77 * pow(abs(omega / peakOmega), -2.5);
//    }
//    return 6.97 * pow(abs(omega / peakOmega), 5.0);
//}
//
//fn DirectionSpectrum(theta: f32, omega: f32, spectrum: SpectrumParameters) -> f32{
//    let s = SpreadPower(omega, spectrum.peakOmega) + 16 * tanh(min(omega/ spectrum.peakOmega,20.0)) * spectrum.swell * spectrum.swell;
//    return mix(2.0 / PI * cos(theta) * cos(theta), Cosine2s(theta - spectrum.angle, s), spectrum.spreadBlend);
//}
//
//fn TMACorrection(omega: f32) -> f32{
//    let omegaH = omega * sqrt(_Depth / _Gravity);
//    if(omegaH <= 1.0){
//        return 0.5 * omegaH * omegaH;
//    }
//    if(omegaH < 2.0){
//        return 1.0 - 0.5 * (2.0 - omegaH) * (2.0 - omegaH);
//    }
//    return 1.0;
//}
//
//fn JONSWAP(omega: f32, spectrum: SpectrumParameters) -> f32 {
//    //select(falseValue, trueValue, condition)
//    let sigma = select(0.09, 0.07, omega <= spectrum.peakOmega);
//
//    let r = exp(-(omega - spectrum.peakOmega) * (omega - spectrum.peakOmega) / 2.0 / sigma / sigma /spectrum.peakOmega / spectrum.peakOmega);
//
//    let oneOverOmega = 1.0 / omega;
//
//    let peakOmegaOverOmega = spectrum.peakOmega / omega;
//
//    return spectrum.scale * TMACorrection(omega) * spectrum.alpha * _Gravity * _Gravity
//        * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega
//        * exp(-1.5 * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega)
//        * pow(abs(spectrum.gamma), r);
//
//}

@compute @workgroup_size(8, 8, 1)
fn CS_InitializeSpectrum(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {

      textureStore(_InitialSpectrumTextures, id.xy, 0, vec4(1.0,1.0,0.0,1.0));
//    var seed = id.x + _N * id.y + _N;
//    seed += _Seed;
//
//    let lengthScales = array<f32,3>(_LengthScale0, _LengthScale1, _LengthScale2);
//
//    for(let i = 0; i<3; i++){
//        let halfN = _N/2;
//        let deltaK = 2.0 * PI/lengthScales[i];
//        let K = (id.xy - halfN) * deltaK;
//        let kLength = length(K);
//
//       seed += i + hash(seed) * 10;
//       let uniformRandSamples = vec4(hash(seed), hash(seed*2), hash(seed*3), hash(seed*4));
//       let gauss1 = UniformToGaussian(uniformRandSamples.x, uniformRandSamples.y);
//       let gauss2 = UniformToGaussian(uniformRandSamples.z, uniformRandSamples.w);
//       if(_Lowcutoff <= kLength && kLength <= HeighCutoff){
//           let kAngle = atan2(K.y, K.x);
//           let omega = Dispersion(kLength);
//           let dOmegadk = DispersionDerivative(kLength);
//
//           var spectrum = JonSWAP(omega, _Spectrums[i * 2]) * DirectionSpectrum(kAngle, omega, _Spectrums[i * 2]) * ShortWavesFade(kLength, _Spectrums[i * 2]);
//
//           if(_Spectrums[i * 2 + 1].scale > 0){
//                spectrum += JonSWAP(omega, _Spectrums[i * 2 +1]) * DirectionSpectrum(kAngle, omega, _Spectrums[i * 2 +1]) * ShortWavesFade(kLength, _Spectrums[i * 2 +1]);
//           }
//           let value = vec4(vec2(gauss2.x, gauss1.y) * sprt(2 * spectrum * abs(dOmegadk) / kLength  * deltaK * deltaK), 0.0,0.0);
//           textureStore(_InitialSpectrumTextures, id.xy, i, value);
//
//       }else{
//
//           textureStore(_InitialSpectrumTextures, id.xy, i, vec4(0.0));
//       }


//    }
}
