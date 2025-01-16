#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings as view_bindings

const PI = 3.141592;
const _roughness = 0.1;
const F0 = 1.0;
const specular_color = vec3(0.1, 0.1, 0.1);
const light_dir = vec3(0.0, 1.0, 5.0);

@group(2) @binding(0) var<uniform> scatter_color: vec4<f32>;
@group(2) @binding(11) var<uniform> sun_color: vec4<f32>;
@group(2) @binding(12) var<uniform> ambient_color: vec4<f32>;
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

struct DispFoam {
    displacement: vec3<f32>,
    foam: vec3<f32>,
}

fn get_displacement_and_foam(current_pos: vec2<f32>) -> DispFoam {
    let displacement1 = textureSample(displacement_texture, textureSampler, current_pos / tile_1, 0);
    let displacement2 = textureSample(displacement_texture, textureSampler, current_pos / tile_2, 1);
    let displacement3 = textureSample(displacement_texture, textureSampler, current_pos / tile_3, 2);
    let disp = displacement1.xyz * tile_1 + displacement2.xyz * tile_2 + displacement3.xyz * tile_3;
    let foam = vec3(displacement1.w * foam_1, displacement2.w * foam_2, displacement3.w * foam_3);
    return DispFoam(disp, foam);

}

fn get_displacement(current_pos: vec2<f32>) -> vec3<f32> {
    var displacement = textureSample(displacement_texture, textureSampler, current_pos / tile_1, 0).xyz * tile_1;
    displacement += textureSample(displacement_texture, textureSampler, current_pos / tile_2, 1).xyz * tile_2;
    displacement += textureSample(displacement_texture, textureSampler, current_pos / tile_3, 2).xyz * tile_3;
    return displacement;
}
fn slope_factor(distance: f32, near: f32, far: f32) -> f32{
    let a = clamp( (-1.0/(far-near)) * (distance - far) , 0.0,1.0);
    return a * a;
}

fn get_slope(current_pos: vec2<f32>, distance: f32) -> vec2<f32> {
    var slope = textureSample(slope_texture, textureSampler, current_pos / tile_1, 0).xy * slope_factor(distance, 500.0, 20000.0);
    slope += textureSample(slope_texture, textureSampler, current_pos / tile_2, 1).xy * slope_factor(distance, 400.0, 1200.0);
    slope += textureSample(slope_texture, textureSampler, current_pos / tile_3, 2).xy * slope_factor(distance, 200.0, 800.0);
    return slope;
}


fn CalcD(NdotH: f32, roughness: f32) -> f32 {
    //    let m_squared: f32 = roughness * roughness;
    //    let a = NdotH * m_squared;
    //    let b = (1 + NdotH * NdotH * (m_squared * m_squared - 1));
    //    return a / (4* b * b);

    let m_squared: f32 = roughness * roughness;
    let r1: f32 = 1. / (4. * m_squared * pow(NdotH, 4.));
    let r2: f32 = (NdotH * NdotH - 1.) / (m_squared * NdotH * NdotH);
    let D: f32 = r1 * exp(r2);
    return D;
}


fn CookTorrance(materialSpecularColor: vec3<f32>, normal: vec3<f32>, lightDir: vec3<f32>, viewDir: vec3<f32>, lightColor: vec3<f32>, roughness: f32) -> vec3<f32> {
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
        let D = CalcD(NdotH, roughness);
        //geometric shadowing
        let two_NdotH: f32 = 2. * NdotH;
        let g1: f32 = two_NdotH * NdotV / VdotH;
        let g2: f32 = two_NdotH * NdotL / VdotH;
        let G: f32 = min(1., min(g1, g2));
        Rs = max(F * D * G / (PI * NdotL * NdotV), 0.0);
    }
    return lightColor * materialSpecularColor * Rs;
}

fn dot_clamped(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return max(0.0, dot(a, b));

}

fn color_from_position(current_pos: vec2<f32>, camera_position: vec3<f32>, view_dir: vec3<f32> ) -> vec4<f32>{
    //    let disp = textureSample(displacement_texture, textureSampler, current_pos /tile_1, 0);
    let disp_foam = get_displacement_and_foam(current_pos);
    let light = normalize(light_dir);
     let water_pos = vec3(current_pos.x, 0.0, current_pos.y) + disp_foam.displacement;
     let distance = dot(water_pos - camera_position, view_dir);
    let slope = get_slope(current_pos, distance);
    var normal = normalize(vec3(-slope.x, 1.0, -slope.y));

    let brightness = dot_clamped(normal, light);
    var r = reflect(view_dir, normal);

    r.z = -r.z;
    let reflected = textureSample(skybox, textureSampler, r);

    let F: f32 = pow(1. - dot(-view_dir, normal), 5.);

    let foam_color = vec3(1.0, 1.0, 1.0);
    let foam_amount = clamp(disp_foam.foam.x + disp_foam.foam.y + disp_foam.foam.z, 0.0, 1.0);
    var a = _roughness + 1.0 * foam_amount;
    let r_factor = slope_factor(distance, 0.0, 20000.0);
    a = a +  (1.0 - r_factor) * 1.0;

    let H = max(0.0, disp_foam.displacement.y);

    let k1 = 2.0 * H * pow(dot_clamped(light, view_dir), 4.0) * pow(0.5 - 0.5 * dot_clamped(light, normal), 3.0);
    let k2 = pow(dot_clamped(-view_dir, normal), 2.0);
    let k3 = brightness;

    let Half: vec3<f32> = normalize(light_dir - view_dir);
    let NdotH: f32 = dot_clamped(normal, Half);
    var D = 0.0;
    if (NdotH > 0) {
        D = CalcD(NdotH, a);
    }

    var scatter = (k1 + k2) * scatter_color * sun_color * (1.0 / (1.0 + D));
    var foam = (k1 + k2) * foam_color * sun_color.xyz * (1.0 / (1.0 + D));
    foam += k3 * foam_color * sun_color.xyz + ambient_color.xyz * foam_color;
    scatter += k3 * scatter_color * sun_color + ambient_color * scatter_color;

    let water_color = mix(scatter.xyz, foam, foam_amount);

    //        return vec4(tile_1);
    return vec4((F * (1.0 - foam_amount)) * reflected.xyz + (1.0 - (F * (1.0 - foam_amount))) * water_color + CookTorrance(specular_color, normal, light, -view_dir, sun_color.xyz, a), select(0.0,1.0,distance > 0.0));

}

const slopeUB: f32 = 2.0;
const g: f32 = sin(atan(1.0/ slopeUB));

fn sde(displacement_h: f32, pos_h: f32) -> f32{
    return (pos_h - displacement_h) * g;
}



@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    //    return textureSample(displacement_texture, textureSampler, mesh.uv, 0);
    let pos = mesh.world_position.xyz;

    //    let og_point = vec3(mesh.uv.x, 1.0, mesh.uv.y);
    let og_point = vec3(pos.x, pos.y, pos.z);

    let camera_position = vec3<f32>(view_bindings::view.world_from_view[3].xyz);
    let view_dir = normalize(pos - camera_position);
    //    return textureSample(displacement_texture, textureSampler, mesh.uv + view_dir.xz/(-view_dir.y * 2), 0);

    var current_pos = og_point.xz;
    var difference = 100.0;

    var t = 0.0;

    //marching

    var i = 0;
    while(i< 150 && difference > 0.1) {
        let displacement = get_displacement(current_pos);
        let displace_point = vec3(current_pos.x, 0, current_pos.y) + displacement;
        let s = (displacement.y - og_point.y) / view_dir.y;
        let hit = og_point + s * view_dir;
        let new_pos = hit - displacement;
        let new_coords = vec2(new_pos.x, new_pos.z);
        let diff = new_coords - current_pos;
//        let h = (og_point + view_dir * (length(current_pos - og_point.xz)/ length(view_dir.xz))).y;
//        let estimate = sde(displacement.y, h);

//       let point = og_point + view_dir * t;



        current_pos = current_pos + diff * 0.2;

        difference = length(diff);
//        current_pos = current_pos + normalize(diff) * clamp(length(diff) * 0.2, 0.0, 2.0);
    //        let op = displace_point - og_point;
    //    //    let x = dot(op, view_dir) - view_dir;
    //        let x = -op + view_dir * dot(view_dir, op);
    //        current_pos = current_pos + normalize(x.xz) * length(x) * factor;
        i = i+1;

    }


//     let water_pos = vec3(current_pos.x, 0.0, current_pos.y) + disp_foam.displacement;
     let distance = dot(vec3(current_pos.x, 0.0, current_pos.y) - camera_position, view_dir);
     var col = vec4(0.0);
     for(var i = -2; i < 3; i++){
        for(var j = -2; j< 3; j++){
            col += color_from_position(current_pos + vec2(f32(i), f32(j)) * distance * 0.0001, camera_position, view_dir)/ 25.0;
        }
     }
//     return vec4(vec3(f32(i)/ 150.0),1.0);
     return col;

    //    return vec4(textureSample(displacement_texture, textureSampler, current_pos /10.0, 0).y/2.0 + 0.2);

    //shader stuff

//    return vec4(water_pos, 1.0);

//    return vec4(1.0) * dot(normal, light);
//    return vec4(og_point, 1.0);
//return vec4(factor);

// get vector & distance to view line from displaced position
// go in that direction from the original pointer
// repeat a few times
// stonks
}
