#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world}
#import bevy_pbr::forward_io::Vertex
#import bevy_pbr::mesh_view_bindings::globals
#import bevy_pbr::mesh_view_bindings as view_bindings

const PI = 3.141592;
const roughness = 0.05;
const F0 = 1.0;
const ambient = vec3(0.02, 0.06, 0.08);
//const ambient = vec3(0.0, 0.0, 0.0);
const specular_color = vec3(0.1, 0.1, 0.1);
const light_color = vec3(1.0, 0.8, 0.2);
//const light_color = vec3(0.0, 0.00, 0.0);
const light_dir = vec3(0.3, 1.0, -5.0);

@group(2) @binding(0) var<uniform> material_color: vec4<f32>;
@group(2) @binding(1) var textureSampler: sampler;
@group(2) @binding(2) var skybox: texture_cube<f32>;
@group(2) @binding(3) var displacement_texture: texture_2d_array<f32>;
@group(2) @binding(4) var slope_texture: texture_2d_array<f32>;

@group(2) @binding(5) var<uniform> tile_1: f32;
@group(2) @binding(6) var<uniform> tile_2: f32;
@group(2) @binding(7) var<uniform> tile_3: f32;

@group(2) @binding(8) var<uniform> foam_1: f32;
@group(2) @binding(9) var<uniform> foam_2: f32;
@group(2) @binding(10) var<uniform> foam_3: f32;

//const waveA = array<f32, 4>(0.01, 0.2, 0.4, 0.5);
//const waveX = array<f32, 4>(-1.8, 1.0, -0.4, 0.1);
//const waveZ = array<f32, 4>(0.5, 0.2, 0.2, -0.2);
//const waveSpeed = array<f32, 4>(2.3, -1.3, 5.8, 0.1);

const waves = array<vec4<f32>, 30>(vec4(0.99, 0.12, 0.08, 12.3), vec4(-0.85, 0.52, 0.15, 5.4), vec4(0.37, -0.93, 0.12, 18.7), vec4(0.71, -0.71, 0.22, 8.9), vec4(-0.45, -0.89, 0.18, 25.2), vec4(0.95, 0.31, 0.10, 4.5), vec4(-0.62, 0.78, 0.05, 10.1), vec4(0.20, -0.98, 0.11, 15.8), vec4(-0.99, 0.12, 0.14, 6.3), vec4(0.52, 0.85, 0.09, 30.0), vec4(0.85, -0.52, 0.25, 40.5), vec4(-0.37, 0.93, 0.21, 22.7), vec4(0.45, 0.89, 0.07, 12.9), vec4(-0.71, -0.71, 0.18, 28.6), vec4(0.62, -0.78, 0.19, 35.4), vec4(-0.95, -0.31, 0.11, 9.2), vec4(0.99, -0.12, 0.23, 45.1), vec4(-0.20, 0.98, 0.13, 7.8), vec4(0.71, 0.71, 0.16, 11.4), vec4(-0.52, -0.85, 0.24, 39.2), vec4(0.37, 0.93, 0.20, 17.8), vec4(-0.45, -0.89, 0.09, 5.6), vec4(0.85, -0.52, 0.12, 21.3), vec4(-0.99, 0.12, 0.06, 8.5), vec4(0.62, 0.78, 0.08, 14.6), vec4(-0.71, -0.71, 0.14, 19.5), vec4(0.20, 0.98, 0.22, 33.8), vec4(-0.37, 0.93, 0.19, 27.6), vec4(0.52, -0.85, 0.10, 6.9), vec4(-0.95, 0.31, 0.17, 12.2));

struct waterInfo {
    dist: vec3<f32>,
    tangent: vec3<f32>,
    binormal: vec3<f32>,
}

fn GerstWave(wave: vec4<f32>, p: vec3<f32>) -> waterInfo {
    let steepness = wave.z;
    let wavelength = wave.w;
    let k = 2.0 * PI / wavelength;
    //    let c = sqrt(9.8/k);
    let c = sqrt(2.8 / k);
    let d = normalize(wave.xy);
    let f = k * dot(d, p.xz) - c * globals.time;
    let a = steepness / k;

    let tangent = vec3(-d.x * d.x * (steepness * sin(f)), d.x * (steepness * cos(f)), -d.x * d.y * (steepness * sin(f)));

    let binormal = vec3(-d.x * d.y * (steepness * sin(f)), d.y * (steepness * cos(f)), -d.y * d.y * (steepness * sin(f)));

    let dist = vec3(d.x * (a * cos(f)), a * sin(f), d.y * (a * cos(f)));
    return waterInfo(dist, tangent, binormal);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
}
;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let w_pos = mesh_position_local_to_world(get_world_from_local(vertex.instance_index),vec4<f32>(vertex.position, 1.0),).xyz;
    var pos = vertex.position +  textureSampleLevel(displacement_texture, textureSampler, w_pos.xz / tile_1, 0, 0.0).xyz * tile_1;
    pos += textureSampleLevel(displacement_texture, textureSampler, w_pos.xz / tile_2, 1, 0.0).xyz * tile_2;
    pos += textureSampleLevel(displacement_texture, textureSampler, w_pos.xz / tile_3, 2, 0.0).xyz * tile_3;
//    for (var i = 0; i < 0; i++) {
//
//        let info = GerstWave(waves[i], vertex.position);
//        p += info.dist;
//    }
    //    let pos =vertex.position + displacement(vertex.position);

    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(vertex.instance_index),
        vec4<f32>(pos, 1.0),
    );

    out.world_position = vec4<f32>(w_pos, 0.0);
    return out;
}


fn CookTorrance(materialDiffuseColor: vec3<f32>, materialSpecularColor: vec3<f32>, normal: vec3<f32>, lightDir: vec3<f32>, viewDir: vec3<f32>, lightColor: vec3<f32>) -> vec3<f32> {
    let NdotL: f32 = max(0., dot(normal, lightDir));
    var Rs: f32 = 0.;
    if (NdotL > 0) {
        let H: vec3<f32> = normalize(lightDir + viewDir);
        let NdotH: f32 = max(0., dot(normal, H));
        let NdotV: f32 = max(0., dot(normal, viewDir));
        let VdotH: f32 = max(0., dot(lightDir, H));
        //Fresnel
        var F: f32 = pow(1. - VdotH, 5.);
        F = F * (1. - F0);
        F = F + (F0);
        //microfacet
        let m_squared: f32 = roughness * roughness;
        let r1: f32 = 1. / (4. * m_squared * pow(NdotH, 4.));
        let r2: f32 = (NdotH * NdotH - 1.) / (m_squared * NdotH * NdotH);
        let D: f32 = r1 * exp(r2);
        //geometric shadowing
        let two_NdotH: f32 = 2. * NdotH;
        let g1: f32 = two_NdotH * NdotV / VdotH;
        let g2: f32 = two_NdotH * NdotL / VdotH;
        let G: f32 = min(1., min(g1, g2));
        Rs = max(F * D * G / (PI * NdotL * NdotV),0.0);
    }
    return materialDiffuseColor * lightColor * NdotL + lightColor * materialSpecularColor * Rs;
}


@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    //    let pos = mesh.world_position.xyz + displacement(mesh.world_position.xyz);
    //    let tangent = xDer(mesh.world_position.xyz);
    //    let binormal = zDer(mesh.world_position.xyz);
    //    let normal = -normalize(cross(tangent, binormal));

    //    let info =  GerstWave(vec4(1.0,1.0,0.25,60.0), mesh.world_position.xyz);
    //    let info2 =  GerstWave(vec4(1.0,0.6,0.25,31.0), mesh.world_position.xyz);
    //    let info3 =  GerstWave(vec4(1.0,1.3,0.25,18.0), mesh.world_position.xyz);
    var slope = textureSample(slope_texture, textureSampler, mesh.world_position.xz /tile_1, 0).xy;
    slope += textureSample(slope_texture, textureSampler, mesh.world_position.xz /tile_2, 1).xy;
    slope += textureSample(slope_texture, textureSampler, mesh.world_position.xz /tile_3, 2).xy;
    slope *= 10.0;

    let displacement1 =  textureSample(displacement_texture, textureSampler, mesh.world_position.xz / tile_1, 0) ;
    let displacement2 = textureSample(displacement_texture, textureSampler, mesh.world_position.xz  / tile_2, 1);
    let displacement3 = textureSample(displacement_texture, textureSampler, mesh.world_position.xz / tile_3, 2);
    let displacement = (displacement1* tile_1 + displacement2 * tile_2 + displacement3 * tile_3).xyz;


    let p = mesh.world_position.xyz + displacement;

//    var tangent = vec3(1.0, slope.x, 0.0);
//    var binormal = vec3(0.0, slope.y, 1.0);
//    var tangent = vec3(1.0, 0, 0.0);
//    var binormal = vec3(0.0, 0, 1.0);
//    for (var i = 0; i < 30; i++) {
//
//        let info = GerstWave(waves[i], mesh.world_position.xyz);
//        p += info.dist;
//        tangent += info.tangent;
//        binormal += info.binormal;
//    }

//    var normal = normalize(cross(binormal, tangent));
//    var normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    var normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    normal = normalize(mix(normal, vec3(0.0,1.0,0.0), 0.0));
//    let normal: vec3<f32> = normalize(textureSample(slope, textureSampler, mesh.world_position.xy /256.0, 0).xy);

    let light = normalize(light_dir);
    let brightness = max(dot(normal, light),0.0);

    let camera_position = vec3<f32>(view_bindings::view.world_from_view[3].xyz);
    let view_dir = normalize(p - camera_position);

    var r = reflect(view_dir, normal);
    r.z = -r.z;
    let reflected = textureSample(skybox, textureSampler, r);

    let F: f32 = pow(1. - dot(-view_dir, normal), 5.);
    //    return reflected;
    //    return vec4(mesh.world_position.xyz, 1.0);

//    return vec4(textureSample(displacement, skyboxSampler, mesh.world_position.xz / 256.0, 0).xyz,1.0);
//    return vec4(slope,0.0,1.0);

//    return vec4(normal, 1.0);


//
    let foam_color = vec3(2.0, 2.0,2.0);
    let foam_amount = (displacement1.w * foam_1) + (displacement2.w * foam_2) + (displacement3.w * foam_3);

//    let water_color = material_color.xyz +  vec3(1.0,1.0,1.0) * displacement.w;
//    let water_color = foam_color * displacement.w + material_color.xyz*(1.0 - displacement.w);
    let water_color = mix(material_color.xyz, foam_color, clamp(foam_amount,0.0,1.0));

    return vec4( 1.0 * F * reflected.xyz + (water_color) * ambient + brightness * CookTorrance(water_color, specular_color, normal, light, -view_dir, light_color), 1.0);
//    return vec4(material_color.xyz * ambient + brightness * CookTorrance(material_color.xyz, specular_color, normal, light, -view_dir, light_color),1.0);
//    return vec4(material_color.xyz * ambient + brightness * material_color.xyz,1.0);
//       return vec4(brightness * material.color.xyz, 1.0);
//    return vec4(binormal, 1.0);
//    return material.color;
//    return material.color * textureSample(material_color_texture, material_color_sampler, mesh.uv);
}
