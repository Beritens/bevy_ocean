mod water;

use crate::water::{SpectrumParameters, WaterPlugin, WaterResource};
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::Skybox;
use bevy::image::{
    CompressedImageFormats, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
};
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, Extent3d, FilterMode, SamplerDescriptor, ShaderRef, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, TouchControls};
use std::f32::consts::TAU;

const SHADER_ASSET_PATH: &str = "shaders/custom_material.wgsl";

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin {
                default_sampler: ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    address_mode_w: ImageAddressMode::Repeat,
                    mag_filter: ImageFilterMode::Linear,
                    ..Default::default()
                },
            }),
            MaterialPlugin::<CustomMaterial>::default(),
        ))
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(WaterPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update_time)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    // Create and save a handle to the mesh.
    let skybox_handle = asset_server.load("textures/Ryfjallet_cubemap_bc7.ktx2");
    // let skybox_handle = asset_server.load("textures/cubemap.ktx2");

    // camera
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera::default(),
        Skybox {
            brightness: 1000.0,
            image: skybox_handle.clone(),
            ..default()
        },
    ));

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

    displacement_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

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

    slope_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let image0 = images.add(initial_spectrum_texture);
    let image1 = images.add(spectrum_texture);
    let image2 = images.add(displacement_texture);
    let image3 = images.add(slope_texture);

    let spectrum: SpectrumParameters = SpectrumParameters {
        scale: 0.001,
        angle: 3.14,
        spreadBlend: 0.4,
        swell: 0.2,
        alpha: 0.01,
        peakOmega: 5.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum1: SpectrumParameters = SpectrumParameters {
        scale: 0.001,
        angle: 90.0,
        spreadBlend: 0.5,
        swell: 2.0,
        alpha: 0.01,
        peakOmega: 6.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum2: SpectrumParameters = SpectrumParameters {
        scale: 0.001,
        angle: 1.6,
        spreadBlend: 0.8,
        swell: 0.00,
        alpha: 0.01,
        peakOmega: 8.0,
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
        _seed: 1,
        _LengthScale0: 10.0,
        _LengthScale1: 8.0,
        _LengthScale2: 2.0,
        _LowCutoff: 0.0001,
        _HighCutoff: 1000.0,
        _Gravity: 9.8,
        _RepeatTime: 20.0,
        _FrameTime: 0.0,
        _Lambda: Vec2 { x: 0.8, y: 0.8 },
        _Spectrums: spectrums,
        _SpectrumTextures: image1.clone(),
        _InitialSpectrumTextures: image0.clone(),
        _DisplacementTextures: image2.clone(),
        _SlopeTextures: image3.clone(),
        _Depth: 10000.0,
        _FourierTarget: image1.clone(),
        _FoamBias: 1.0,
        _FoamDecayRate: 2.0,
        _FoamAdd: 10.0,
        _FoamThreshold: 0.0,
    };

    commands.insert_resource(water_resource);
    let mat = materials.add(CustomMaterial {
        color: LinearRgba::new(0.0, 0.2, 1.0, 1.0),
        skybox_texture: skybox_handle.clone(),
        displacement: image2.clone(),
        slope: image3.clone(),
        tile_1: 2024.0,
        tile_2: 600.0,
        tile_3: 120.0,
        foam_1: 0.1,
        foam_2: 1.0,
        foam_3: 0.4,
        alpha_mode: AlphaMode::Opaque,
    });

    let plane: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(1000.0, 1000.0)
            .subdivisions(1000),
    ));

    let plane2: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(1000.0, 1000.0)
            .subdivisions(500),
    ));

    let plane3: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(3000.0, 3000.0)
            .subdivisions(100),
    ));

    for i in -5..6 {
        for j in -5..6 {
            if i == 0 && j == 0 {
                for k in -1..2 {
                    for l in -1..2 {
                        if k == 0 && l == 0 {
                            commands.spawn((
                                Mesh3d(plane.clone()),
                                MeshMaterial3d(mat.clone()),
                                Transform::from_xyz(i as f32 * 1000.0, 0.0, j as f32 * 1000.0),
                            ));
                        } else {
                            commands.spawn((
                                Mesh3d(plane2.clone()),
                                MeshMaterial3d(mat.clone()),
                                Transform::from_xyz(k as f32 * 1000.0, 0.0, l as f32 * 1000.0),
                            ));
                        }
                    }
                }
            } else {
                commands.spawn((
                    Mesh3d(plane3.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_xyz(i as f32 * 3000.0, 0.0, j as f32 * 3000.0),
                ));
            }
        }
    }
}

fn update_time(time: Res<Time>, mut water_resource: ResMut<WaterResource>) {
    water_resource._FrameTime += time.delta_secs() * 0.2;
}

fn generate_custom_mesh() -> Mesh {
    let mut coordinates = Vec::new();
    let mut indices = Vec::new();
    let width = 2000;
    let depth = 2000;
    let spacing = 0.4;
    for z in 0..depth {
        for x in 0..width {
            coordinates.push([
                x as f32 * spacing - (width as f32 / 2.0) * spacing,
                0.0,
                z as f32 * spacing - (depth as f32 / 2.0) * spacing,
            ]);
        }
    }

    for x in 0..width - 1 {
        for z in 0..depth - 1 {
            indices.extend([z * width + x, (z + 1) * (width) + x, z * width + x + 1]);

            indices.extend([
                z * width + x + 1,
                (z + 1) * width + x,
                (z + 1) * width + x + 1,
            ]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, coordinates)
    .with_inserted_indices(Indices::U32(indices))
}

// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    color: LinearRgba,
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
