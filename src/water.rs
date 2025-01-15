use crate::water_compute::{SpectrumParameters, WaterResource};
use crate::Buoyant;
use avian3d::prelude::RigidBody;
use bevy::app::{App, Plugin, PostStartup, PreUpdate, Startup, Update};
use bevy::asset::{Asset, Handle, LoadState, RenderAssetUsages};
use bevy::color::LinearRgba;
use bevy::image::{Image, ImageSampler, ImageSamplerDescriptor};
use bevy::math::ops::tan;
use bevy::math::{Quat, UVec2, Vec3};
use bevy::pbr::{Material, MaterialPlugin, NotShadowCaster};
use bevy::prelude::{
    default, AlphaMode, AssetServer, Assets, Commands, Component, Entity, Mesh, Mesh3d,
    MeshMaterial3d, Meshable, Mut, Plane3d, Query, Res, ResMut, Resource, StandardMaterial, Time,
    Transform, Trigger, TypePath, Vec2, Window, With, Without,
};
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_graph::RenderGraph;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::window::PrimaryWindow;
use crossbeam_channel::{Receiver, Sender};
use wgpu::TextureViewDescriptor;

pub struct WaterPlugin;

const SHADER_ASSET_PATH: &str = "shaders/custom_material.wgsl";
const SHADER_ASSET_PATH_RAY_MARCHING: &str = "shaders/ray_marching.wgsl";

#[derive(Component)]
struct DisplacementReceiver {
    receiver: Receiver<Vec<Vec<Vec3>>>,
}

#[derive(Component)]
struct SlopeReceiver {
    receiver: Receiver<Vec<Vec<Vec2>>>,
}

#[derive(Component)]
pub struct DisplacementImage {
    pub displacement: Vec<Vec<Vec3>>,
}

#[derive(Component)]
pub struct SlopeImage {
    pub slope: Vec<Vec<Vec2>>,
}

impl Plugin for crate::water::WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crate::water_compute::WaterComputePlugin);

        // app.add_plugins(MaterialPlugin::<CustomMaterial>::default());
        app.add_plugins(MaterialPlugin::<MarchMaterial>::default());
        app.add_systems(PostStartup, water_setup);
        app.add_systems(
            PreUpdate,
            (receive_displacement_texture, receive_slope_texture),
        );
        app.add_systems(Update, (update_time, reinterpret_cubemap));
    }
}

fn water_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut march_materials: ResMut<Assets<MarchMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    // Create and save a handle to the mesh.
    let skybox_handle = asset_server.load("textures/skybox.png");
    commands.insert_resource(SkyCubeMap {
        image: skybox_handle.clone(),
        loaded: false,
    });

    let mut initial_spectrum_texture = Image::new_fill(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 3,
        },
        TextureDimension::D2,
        &[
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    initial_spectrum_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let mut spectrum_texture = Image::new_fill(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    spectrum_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let mut displacement_texture = Image::new_fill(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 3,
        },
        TextureDimension::D2,
        &[
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    displacement_texture.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC;

    let mut slope_texture = Image::new_fill(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 3,
        },
        TextureDimension::D2,
        &[255, 255, 255, 255, 255, 255, 255, 255],
        TextureFormat::Rg32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    slope_texture.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC;

    let image0 = images.add(initial_spectrum_texture);
    let image1 = images.add(spectrum_texture);
    let image2 = images.add(displacement_texture);
    let image3 = images.add(slope_texture);

    let spectrum: SpectrumParameters = SpectrumParameters {
        scale: 0.1,
        angle: 1.14,
        spreadBlend: 0.6,
        swell: 1.0,
        alpha: 0.01,
        peakOmega: 2.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum1: SpectrumParameters = SpectrumParameters {
        scale: 0.1,
        angle: 1.14,
        spreadBlend: 1.0,
        swell: 0.8,
        alpha: 0.01,
        peakOmega: 3.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum2: SpectrumParameters = SpectrumParameters {
        scale: 0.1,
        angle: 1.6,
        spreadBlend: 0.8,
        swell: 0.00,
        alpha: 0.01,
        peakOmega: 5.0,
        gamma: 1.0,
        shortWavesFade: 0.005,
    };

    let spectrum_array = [
        spectrum.clone(),
        spectrum.clone(),
        spectrum1.clone(),
        spectrum1.clone(),
        spectrum2.clone(),
        spectrum2.clone(),
    ];

    let spectrums = buffers.add(ShaderStorageBuffer::from(spectrum_array));

    let water_resource = WaterResource {
        _N: 256,
        _seed: 69,
        _LengthScale0: 20.0,
        _LengthScale1: 15.0,
        _LengthScale2: 4.0,
        _LowCutoff: 0.0001,
        _HighCutoff: 1000.0,
        _Gravity: 9.8,
        _RepeatTime: 20.0,
        _FrameTime: 0.0,
        _Lambda: Vec2 { x: 0.5, y: 0.5 },
        _Spectrums: spectrums,
        _SpectrumTextures: image1.clone(),
        _InitialSpectrumTextures: image0.clone(),
        _DisplacementTextures: image2.clone(),
        _SlopeTextures: image3.clone(),
        _Depth: 10000.0,
        _FourierTarget: image1.clone(),
        _FoamBias: 0.9,
        _FoamDecayRate: 0.100000,
        _FoamAdd: 0.20,
        _FoamThreshold: 0.0,
    };

    commands.insert_resource(water_resource);
    let mat = march_materials.add(MarchMaterial {
        scatter_color: LinearRgba::new(0.01, 0.04, 0.06, 1.0),
        sun_color: LinearRgba::new(0.4, 0.4, 0.3, 1.0),
        ambient_color: LinearRgba::new(0.01, 0.01, 0.3, 1.0),
        skybox_texture: skybox_handle.clone(),
        displacement: image2.clone(),
        slope: image3.clone(),
        tile_1: 102.0,
        tile_2: 89.0,
        tile_3: 12.0,
        foam_1: 0.2,
        foam_2: 1.0,
        foam_3: 0.4,
        alpha_mode: AlphaMode::Opaque,
    });
    let plane: Handle<Mesh> =
        meshes.add(Mesh::from(Plane3d::default().mesh().size(10000.0, 10000.0)));

    commands.spawn((
        Mesh3d(plane.clone()),
        MeshMaterial3d(mat.clone()),
        Transform::from_xyz(0.0, 6.0, 0.0).with_rotation(Quat::from_rotation_x(20.0)),
        NotShadowCaster,
    ));

    // let plane: Handle<Mesh> = meshes.add(Mesh::from(
    //     Plane3d::default()
    //         .mesh()
    //         .size(100.0, 100.0)
    //         .subdivisions(1000),
    // ));
    //
    // let plane2: Handle<Mesh> = meshes.add(Mesh::from(
    //     Plane3d::default()
    //         .mesh()
    //         .size(100.0, 100.0)
    //         .subdivisions(500),
    // ));
    //
    // let plane3: Handle<Mesh> = meshes.add(Mesh::from(
    //     Plane3d::default()
    //         .mesh()
    //         .size(300.0, 300.0)
    //         .subdivisions(100),
    // ));
    //
    // for i in -5..6 {
    //     for j in -5..6 {
    //         if i == 0 && j == 0 {
    //             for k in -1..2 {
    //                 for l in -1..2 {
    //                     if k == 0 && l == 0 {
    //                         commands.spawn((
    //                             Mesh3d(plane.clone()),
    //                             MeshMaterial3d(mat.clone()),
    //                             Transform::from_xyz(i as f32 * 100.0, 0.0, j as f32 * 100.0),
    //                             NotShadowCaster,
    //                         ));
    //                     } else {
    //                         commands.spawn((
    //                             Mesh3d(plane2.clone()),
    //                             MeshMaterial3d(mat.clone()),
    //                             Transform::from_xyz(k as f32 * 100.0, 0.0, l as f32 * 100.0),
    //                             NotShadowCaster,
    //                         ));
    //                     }
    //                 }
    //             }
    //         } else {
    //             commands.spawn((
    //                 Mesh3d(plane3.clone()),
    //                 MeshMaterial3d(mat.clone()),
    //                 Transform::from_xyz(i as f32 * 300.0, 0.0, j as f32 * 300.0),
    //                 NotShadowCaster,
    //             ));
    //         }
    //     }
    // }

    let (tx, rx): (Sender<Vec<Vec<Vec3>>>, Receiver<Vec<Vec<Vec3>>>) =
        crossbeam_channel::bounded(300000);

    commands.spawn(Readback::texture(image2.clone())).observe(
        move |trigger: Trigger<ReadbackComplete>| {
            let image_data: &Vec<u8> = &trigger.event().0;
            let displacement_map = convert_readback_data(image_data);

            if (displacement_map[0][0].is_finite()) {
                tx.send(displacement_map).unwrap();
            }
        },
    );

    let (tx2, rx2): (Sender<Vec<Vec<Vec2>>>, Receiver<Vec<Vec<Vec2>>>) =
        crossbeam_channel::bounded(300000);
    commands.spawn(Readback::texture(image3.clone())).observe(
        move |trigger: Trigger<ReadbackComplete>| {
            let image_data: &Vec<u8> = &trigger.event().0;
            let slope_map = convert_readback_data_vec2(image_data);

            if (slope_map[0][0].is_finite()) {
                tx2.send(slope_map).unwrap();
            }
        },
    );
    commands.spawn(DisplacementReceiver { receiver: rx });
    commands.spawn(SlopeReceiver { receiver: rx2 });
}

fn convert_readback_data(data: &[u8]) -> Vec<Vec<Vec3>> {
    let row_bytes = 256 * 4 * 4;
    let aligned_row_bytes = align_byte_size(row_bytes as u32) as usize;

    let mut displacement_map: Vec<Vec<Vec3>> = Vec::with_capacity(256);

    for y in 0..256 {
        let row_start = y * aligned_row_bytes;
        let row_end = row_start + row_bytes.min(data.len().saturating_sub(row_start));

        if row_start >= data.len() {
            break; // Avoid out-of-bounds access
        }

        let row_data = &data[row_start..row_end];
        let row: Vec<Vec3> = row_data
            .chunks(16) // 16 bytes per pixel (RGBA32 float)
            .map(|pixel| {
                let r = f32::from_ne_bytes(pixel[0..4].try_into().unwrap());
                let g = f32::from_ne_bytes(pixel[4..8].try_into().unwrap());
                let b = f32::from_ne_bytes(pixel[8..12].try_into().unwrap());
                Vec3::new(r, g, b) // Convert to Vec3
            })
            .collect();

        displacement_map.push(row);
    }

    displacement_map
}

fn convert_readback_data_vec2(data: &[u8]) -> Vec<Vec<Vec2>> {
    let row_bytes = 256 * 4 * 2;
    let aligned_row_bytes = align_byte_size(row_bytes as u32) as usize;

    let mut displacement_map: Vec<Vec<Vec2>> = Vec::with_capacity(256);

    for y in 0..256 {
        let row_start = y * aligned_row_bytes;
        let row_end = row_start + row_bytes.min(data.len().saturating_sub(row_start));

        if row_start >= data.len() {
            break; // Avoid out-of-bounds access
        }

        let row_data = &data[row_start..row_end];
        let row: Vec<Vec2> = row_data
            .chunks(8) // 16 bytes per pixel (RGBA32 float)
            .map(|pixel| {
                let r = f32::from_ne_bytes(pixel[0..4].try_into().unwrap());
                let g = f32::from_ne_bytes(pixel[4..8].try_into().unwrap());
                // let b = f32::from_ne_bytes(pixel[8..12].try_into().unwrap());
                Vec2::new(r, g) // Convert to Vec3
            })
            .collect();

        displacement_map.push(row);
    }

    displacement_map
}

fn receive_displacement_texture(
    mut commands: Commands,
    receiver_query: Query<&DisplacementReceiver>,
    mut displacement_image_query: Query<&mut DisplacementImage>,
) {
    if let Ok(receiver) = receiver_query.get_single() {
        let result = receiver.receiver.try_recv();
        if let Ok(result) = result {
            if let Ok(mut displacement_image) = displacement_image_query.get_single_mut() {
                displacement_image.displacement = result.clone();
            } else {
                commands.spawn(DisplacementImage {
                    displacement: result.clone(),
                });
            }
        }
    }
}

fn receive_slope_texture(
    mut commands: Commands,
    receiver_query: Query<&SlopeReceiver>,
    mut slope_image_query: Query<&mut SlopeImage>,
) {
    if let Ok(receiver) = receiver_query.get_single() {
        let result = receiver.receiver.try_recv();
        if let Ok(result) = result {
            if let Ok(mut slope_image) = slope_image_query.get_single_mut() {
                slope_image.slope = result.clone();
            } else {
                commands.spawn(SlopeImage {
                    slope: result.clone(),
                });
            }
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    scatter_color: LinearRgba,
    #[uniform(11)]
    sun_color: LinearRgba,
    #[uniform(12)]
    ambient_color: LinearRgba,
    #[sampler(1)]
    #[texture(2, dimension = "cube")]
    skybox_texture: Handle<Image>,

    #[texture(3, dimension = "2d_array")]
    displacement: Handle<Image>,

    #[texture(4, dimension = "2d_array")]
    slope: Handle<Image>,
    #[uniform(5)]
    tile_1: f32,
    #[uniform(6)]
    tile_2: f32,
    #[uniform(7)]
    tile_3: f32,
    #[uniform(8)]
    foam_1: f32,
    #[uniform(9)]
    foam_2: f32,
    #[uniform(10)]
    foam_3: f32,

    alpha_mode: AlphaMode,
}

impl Material for CustomMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct MarchMaterial {
    #[uniform(0)]
    scatter_color: LinearRgba,
    #[uniform(11)]
    sun_color: LinearRgba,
    #[uniform(12)]
    ambient_color: LinearRgba,
    #[sampler(1)]
    #[texture(2, dimension = "cube")]
    skybox_texture: Handle<Image>,

    #[texture(3, dimension = "2d_array")]
    displacement: Handle<Image>,

    #[texture(4, dimension = "2d_array")]
    slope: Handle<Image>,
    #[uniform(5)]
    tile_1: f32,
    #[uniform(6)]
    tile_2: f32,
    #[uniform(7)]
    tile_3: f32,
    #[uniform(8)]
    foam_1: f32,
    #[uniform(9)]
    foam_2: f32,
    #[uniform(10)]
    foam_3: f32,

    alpha_mode: AlphaMode,
}

impl Material for MarchMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH_RAY_MARCHING.into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}

pub(crate) fn align_byte_size(value: u32) -> u32 {
    value + (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - (value % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
}

pub fn get_displacement(n: i32, scale: f32, image: &Vec<Vec<Vec3>>, pos: Vec2) -> Vec3 {
    let scaled = pos / scale * n as f32;
    let x1y1 = Vec2::new(scaled.x.floor(), scaled.y.floor());
    let x2y1 = Vec2::new(scaled.x.ceil(), scaled.y.floor());
    let x1y2 = Vec2::new(scaled.x.floor(), scaled.y.ceil());
    let x2y2 = Vec2::new(scaled.x.ceil(), scaled.y.ceil());
    let x_percent = scaled.x - x1y1.x;
    let y_percent = scaled.y - x1y1.y;
    let displacement_x1_y1 = get_displacement_at_coord(scale, image, get_coords(n, x1y1));
    let displacement_x2_y1 = get_displacement_at_coord(scale, image, get_coords(n, x2y1));
    let displacement_x1_y2 = get_displacement_at_coord(scale, image, get_coords(n, x1y2));
    let displacement_x2_y2 = get_displacement_at_coord(scale, image, get_coords(n, x2y2));
    return x_percent * y_percent * displacement_x2_y2
        + (1.0 - x_percent) * y_percent * displacement_x1_y2
        + x_percent * (1.0 - y_percent) * displacement_x2_y1
        + (1.0 - x_percent) * (1.0 - y_percent) * displacement_x1_y1;
}

pub fn get_normal(n: i32, scale: f32, image: &Vec<Vec<Vec2>>, pos: Vec2) -> Vec3 {
    let slope = get_slope(n, scale, image, pos);
    return Vec3::new(-slope.x, 1.0, -slope.y).normalize();
}
pub fn normal_from_slope(slope: Vec2) -> Vec3 {
    return Vec3::new(-slope.x, 1.0, -slope.y).normalize();
}

pub fn get_slope(n: i32, scale: f32, image: &Vec<Vec<Vec2>>, pos: Vec2) -> Vec2 {
    let scaled = pos / scale * n as f32;
    let x1y1 = Vec2::new(scaled.x.floor(), scaled.y.floor());
    let x2y1 = Vec2::new(scaled.x.ceil(), scaled.y.floor());
    let x1y2 = Vec2::new(scaled.x.floor(), scaled.y.ceil());
    let x2y2 = Vec2::new(scaled.x.ceil(), scaled.y.ceil());
    let x_percent = scaled.x - x1y1.x;
    let y_percent = scaled.y - x1y1.y;
    let displacement_x1_y1 = get_slope_at_coord(scale, image, get_coords(n, x1y1));
    let displacement_x2_y1 = get_slope_at_coord(scale, image, get_coords(n, x2y1));
    let displacement_x1_y2 = get_slope_at_coord(scale, image, get_coords(n, x1y2));
    let displacement_x2_y2 = get_slope_at_coord(scale, image, get_coords(n, x2y2));
    return x_percent * y_percent * displacement_x2_y2
        + (1.0 - x_percent) * y_percent * displacement_x1_y2
        + x_percent * (1.0 - y_percent) * displacement_x2_y1
        + (1.0 - x_percent) * (1.0 - y_percent) * displacement_x1_y1;
}

fn get_displacement_at_coord(scale: f32, image: &Vec<Vec<Vec3>>, pos: UVec2) -> Vec3 {
    return scale * image[pos.y as usize][pos.x as usize];
}

fn get_slope_at_coord(scale: f32, image: &Vec<Vec<Vec2>>, pos: UVec2) -> Vec2 {
    return image[pos.y as usize][pos.x as usize];
}

// fn get_coords(n: i32, pos: Vec2) -> UVec2 {
//     UVec2::new(modulo((pos.x) as i32, n), modulo((pos.y) as i32, n))
// }
fn get_coords(n: i32, pos: Vec2) -> UVec2 {
    UVec2::new(
        (pos.x as i32 & (n - 1)) as u32,
        (pos.y as i32 & (n - 1)) as u32,
    )
}

pub fn get_adjusted_coords(n: i32, scale: f32, pos: Vec2, image: &Vec<Vec<Vec3>>) -> Vec2 {
    let mut adjustedPos: Vec2 = Vec2::from(pos);
    let mut factor = 1.0;
    for i in 0..3 {
        // let coords = get_coords(256, 202.0, adjustedPos, image);
        let displacement = get_displacement(n, scale, image, adjustedPos);
        let newPoint = Vec2::new(
            adjustedPos.x + displacement.x,
            adjustedPos.y + displacement.z,
        );
        // if (pos - newPoint).length_squared() < 0.01 {
        //     return adjustedPos;
        // }
        adjustedPos.x += (pos.x - newPoint.x) * factor;
        adjustedPos.y += (pos.y - newPoint.y) * factor;
        factor *= 0.95
    }
    // return get_coords(n, scale, adjustedPos, image);
    return adjustedPos;
}

fn update_time(
    time: Res<Time>,
    mut water_resource: ResMut<WaterResource>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
) {
    // if let Some(position) = q_windows.single().cursor_position() {
    //     println!("Cursor is inside the primary window, at {:?}", position);
    //     water_resource._FrameTime = position.x * 0.02;
    // } else {
    //     println!("Cursor is not in the game window.");
    // }
    water_resource._FrameTime += time.delta_secs() * 0.2;
}
#[derive(Resource)]
pub struct SkyCubeMap {
    pub image: Handle<Image>,
    pub loaded: bool,
}

pub fn reinterpret_cubemap(
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut cubemap: ResMut<SkyCubeMap>,
) {
    if !cubemap.loaded && asset_server.load_state(&cubemap.image).is_loaded() {
        cubemap.loaded = true;
        let image = images.get_mut(&cubemap.image).unwrap();

        if image.texture_descriptor.array_layer_count() == 1 {
            //6
            image.reinterpret_stacked_2d_as_array(image.height() / image.width());
            image.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..Default::default()
            });
        }
    }
}
