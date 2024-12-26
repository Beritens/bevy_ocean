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
                    // mag_filter: ImageFilterMode::Linear,
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
    let plane: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(1000.0, 1000.0)
            .subdivisions(1000),
    ));
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
        scale: 10.01,
        angle: 90.0,
        spreadBlend: 1.0,
        swell: 2.001,
        alpha: 10.0,
        peakOmega: 0.1,
        gamma: 10.1,
        shortWavesFade: 0.001,
    };

    let spectrum_array = [
        spectrum.clone(),
        spectrum.clone(),
        spectrum.clone(),
        spectrum.clone(),
        spectrum.clone(),
        spectrum.clone(),
    ];

    let spectrums = buffers.add(ShaderStorageBuffer::from(spectrum_array));

    let water_resource = WaterResource {
        _N: 256,
        _seed: 10,
        _LengthScale0: 0.1,
        _LengthScale1: 10.0,
        _LengthScale2: 100.0,
        _LowCutoff: 0.0001,
        _HighCutoff: 1000.0,
        _Gravity: 9.8,
        _RepeatTime: 200.0,
        _FrameTime: 0.0,
        _Lambda: Vec2 { x: 1.0, y: 5.0 },
        _Spectrums: spectrums,
        _SpectrumTextures: image1.clone(),
        _InitialSpectrumTextures: image0.clone(),
        _DisplacementTextures: image2.clone(),
        _SlopeTextures: image3,
        _Depth: 1000000.0,
        _FourierTarget: image1.clone(),
        _FoamBias: 0.0,
        _FoamDecayRate: 2.0,
        _FoamAdd: 1.0,
        _FoamThreshold: 0.0,
    };

    commands.insert_resource(water_resource);

    commands.spawn((
        Mesh3d(plane),
        MeshMaterial3d(materials.add(CustomMaterial {
            color: LinearRgba::new(0.0, 0.2, 1.0, 1.0),
            skybox_texture: skybox_handle.clone(),
            displacement: image2.clone(),
            alpha_mode: AlphaMode::Opaque,
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn update_time(time: Res<Time>, mut water_resource: ResMut<WaterResource>) {
    water_resource._FrameTime += time.delta_secs() * 0.1;
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
