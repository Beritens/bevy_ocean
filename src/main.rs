mod water;

use crate::water::{SpectrumParameters, WaterPlugin, WaterResource};
use bevy::asset::saver::ErasedAssetSaver;
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::Skybox;
use bevy::image::{
    CompressedImageFormats, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
};
use bevy::input::mouse::MouseMotion;
use bevy::pbr::NotShadowCaster;
use bevy::prelude::KeyCode::KeyW;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, Extent3d, FilterMode, SamplerDescriptor, ShaderRef, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::window::PrimaryWindow;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, TouchControls};
use crossbeam_channel::{Receiver, Sender};
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
        .add_systems(
            Update,
            (update_time, receive_displacement_texture, swim, movement),
        )
        .run();
}

#[derive(Component)]
struct Buoyant {}

#[derive(Component)]
struct Control {}
#[derive(Component)]
struct DisplacementImage {
    displacement: Image,
}

#[derive(Component)]
struct DisplacementReceiver {
    receiver: Receiver<Image>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut custom_materials: ResMut<Assets<CustomMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    slope_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let image0 = images.add(initial_spectrum_texture);
    let image1 = images.add(spectrum_texture);
    let image2 = images.add(displacement_texture);
    let image3 = images.add(slope_texture);

    let spectrum: SpectrumParameters = SpectrumParameters {
        scale: 0.01,
        angle: 3.14,
        spreadBlend: 0.9,
        swell: 1.0,
        alpha: 0.01,
        peakOmega: 2.0,
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
        _FoamDecayRate: 0.2,
        _FoamAdd: 10.0,
        _FoamThreshold: 0.0,
    };

    commands.insert_resource(water_resource);
    let mat = custom_materials.add(CustomMaterial {
        color: LinearRgba::new(0.0, 0.2, 1.0, 1.0),
        skybox_texture: skybox_handle.clone(),
        displacement: image2.clone(),
        slope: image3.clone(),
        tile_1: 202.0,
        tile_2: 60.0,
        tile_3: 12.0,
        foam_1: 0.9,
        foam_2: 1.0,
        foam_3: 0.4,
        alpha_mode: AlphaMode::Opaque,
    });

    let plane: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(100.0, 100.0)
            .subdivisions(1000),
    ));

    let plane2: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(100.0, 100.0)
            .subdivisions(500),
    ));

    let plane3: Handle<Mesh> = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(300.0, 300.0)
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
                                Transform::from_xyz(i as f32 * 100.0, 0.0, j as f32 * 100.0),
                                NotShadowCaster,
                            ));
                        } else {
                            commands.spawn((
                                Mesh3d(plane2.clone()),
                                MeshMaterial3d(mat.clone()),
                                Transform::from_xyz(k as f32 * 100.0, 0.0, l as f32 * 100.0),
                                NotShadowCaster,
                            ));
                        }
                    }
                }
            } else {
                commands.spawn((
                    Mesh3d(plane3.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_xyz(i as f32 * 300.0, 0.0, j as f32 * 300.0),
                    NotShadowCaster,
                ));
            }
        }
    }

    let (tx, rx): (Sender<Image>, Receiver<Image>) = crossbeam_channel::bounded(300000);

    commands.spawn(Readback::texture(image2.clone())).observe(
        move |trigger: Trigger<ReadbackComplete>| {
            // You probably want to interpret the data as a color rather than a `ShaderType`,
            // but in this case we know the data is a single channel storage texture, so we can
            // interpret it as a `Vec<u32>`
            // let data: Vec<f32> = trigger.event().to_shader_type();
            let row_bytes = 256 * 4 * 4;
            let aligned_row_bytes = align_byte_size(row_bytes as u32) as usize;

            let image_data = &trigger.event().0;
            let image_data = image_data
                .chunks(aligned_row_bytes)
                .take(256usize * 4)
                .flat_map(|row| &row[..row_bytes.min(row.len())])
                .cloned()
                .collect();
            let mut image = Image::new(
                Extent3d {
                    width: 256,
                    height: 256,
                    ..default()
                },
                TextureDimension::D2,
                image_data,
                TextureFormat::Rgba32Float,
                RenderAssetUsages::default(),
            );

            // info!("Image {:?}", image.get_color_at(0, 0));
            // displament_image.displacement = Some(image.clone());
            tx.send(image.clone()).unwrap();
        },
    );
    commands.spawn(DisplacementReceiver { receiver: rx });

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-1.6),
            ..default()
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
    ));

    for x in -10..10 {
        for z in -10..10 {
            commands.spawn((
                Buoyant {},
                Control {},
                Mesh3d(meshes.add(Sphere::new(5.0))),
                MeshMaterial3d(materials.add(Color::WHITE)),
                Transform::from_xyz(10.0 * x as f32, 50.0, 10.0 * z as f32),
            ));
        }
    }
}
pub(crate) fn align_byte_size(value: u32) -> u32 {
    value + (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - (value % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
}

fn modulo(a: i32, b: i32) -> u32 {
    return (((a % b) + b) % b) as u32;
}

fn movement(keys: Res<ButtonInput<KeyCode>>, mut characters: Query<&mut Transform, With<Control>>) {
    let mut input_vector: Vec2 = Vec2::new(0.0, 0.0);
    if (keys.pressed(KeyCode::KeyW)) {
        input_vector.y += 1.0;
    }
    if (keys.pressed(KeyCode::KeyS)) {
        input_vector.y -= 1.0;
    }
    if (keys.pressed(KeyCode::KeyA)) {
        input_vector.x -= 1.0;
    }
    if (keys.pressed(KeyCode::KeyD)) {
        input_vector.x += 1.0;
    }
    for mut transform in characters.iter_mut() {
        transform.translation.x += 2.0 * input_vector.x;
        transform.translation.z -= 2.0 * input_vector.y;
    }
}

fn get_displacement(scale: f32, image: &Image, pos: UVec2) -> Vec3 {
    let col = image.get_color_at(pos.x, pos.y).unwrap().to_linear();
    return Vec3::new(col.red * scale, col.green * scale, col.blue * scale);
}

fn get_coords(n: i32, scale: f32, pos: Vec2, image: &Image) -> UVec2 {
    let coords = UVec2::new(
        modulo((pos.x / scale * (n as f32)) as i32, n),
        modulo((pos.y / scale * (n as f32)) as i32, n),
    );
    return coords;
}

fn get_adjusted_coords(n: i32, scale: f32, pos: Vec2, image: &Image) -> UVec2 {
    let mut adjustedPos: Vec2 = Vec2::from(pos);
    let mut factor = 1.0;
    for i in 0..4 {
        let coords = get_coords(256, 202.0, adjustedPos, image);
        let displacement = get_displacement(202.0, image, coords);
        let newPoint = Vec2::new(
            adjustedPos.x + displacement.x,
            adjustedPos.y + displacement.z,
        );
        adjustedPos.x += (pos.x - newPoint.x) * factor;
        adjustedPos.y += (pos.y - newPoint.y) * factor;
        factor *= 0.9
    }
    return get_coords(n, scale, adjustedPos, image);
}

fn swim(
    mut buoyant_query: Query<&mut Transform, With<Buoyant>>,
    displacement_image_query: Query<(&DisplacementImage)>,
) {
    if let Ok(displacement_image) = displacement_image_query.get_single() {
        for mut transform in buoyant_query.iter_mut() {
            let coords = get_adjusted_coords(
                256,
                202.0,
                Vec2::new(transform.translation.x, transform.translation.z),
                &displacement_image.displacement,
            );
            let displacement = get_displacement(202.0, &displacement_image.displacement, coords);
            transform.translation.y = displacement.y;
        }
    }
}
fn receive_displacement_texture(
    mut commands: Commands,
    displacement_receiver_query: Query<&DisplacementReceiver>,
    displacement_image_query: Query<(&DisplacementImage, Entity)>,
) {
    if let Ok(receiver) = displacement_receiver_query.get_single() {
        let result = receiver.receiver.try_recv();
        if let Ok(result) = result {
            if let Ok((d, e)) = displacement_image_query.get_single() {
                commands.entity(e).despawn();
            }
            commands.spawn(DisplacementImage {
                displacement: result.clone(),
            });
            // let image = result.convert(TextureFormat::Rgba8UnormSrgb).unwrap();
            // let img = image.try_into_dynamic().unwrap();
            //
            // img.save("tmp.png").unwrap();
        }
    }
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
    water_resource._FrameTime += time.delta_secs() * 0.6;
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
