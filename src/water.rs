use crate::water_compute::{SpectrumParameters, WaterResource};
use bevy::app::{App, Plugin, PreUpdate, Startup, Update};
use bevy::asset::{Asset, Handle, RenderAssetUsages};
use bevy::color::LinearRgba;
use bevy::image::Image;
use bevy::math::{UVec2, Vec3};
use bevy::pbr::{Material, MaterialPlugin, NotShadowCaster};
use bevy::prelude::{
    default, AlphaMode, AssetServer, Assets, Commands, Component, Entity, Mesh, Mesh3d,
    MeshMaterial3d, Meshable, Plane3d, Query, Res, ResMut, StandardMaterial, Time, Transform,
    Trigger, TypePath, Vec2, Window, With,
};
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_graph::RenderGraph;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::window::PrimaryWindow;
use crossbeam_channel::{Receiver, Sender};

pub struct WaterPlugin;

const SHADER_ASSET_PATH: &str = "shaders/custom_material.wgsl";

#[derive(Component)]
struct DisplacementReceiver {
    receiver: Receiver<Image>,
}

#[derive(Component)]
pub struct DisplacementImage {
    pub displacement: Image,
}

impl Plugin for crate::water::WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crate::water_compute::WaterComputePlugin);

        app.add_plugins(MaterialPlugin::<CustomMaterial>::default());
        app.add_systems(Startup, water_setup);
        app.add_systems(PreUpdate, receive_displacement_texture);
        app.add_systems(Update, update_time);
    }
}

fn water_setup(
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
        scale: 0.001,
        angle: 3.14,
        spreadBlend: 0.0,
        swell: 1.0,
        alpha: 0.01,
        peakOmega: 0.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum1: SpectrumParameters = SpectrumParameters {
        scale: 0.00,
        angle: 3.14,
        spreadBlend: 0.9,
        swell: 1.0,
        alpha: 0.01,
        peakOmega: 2.0,
        gamma: 10.1,
        shortWavesFade: 0.005,
    };

    let spectrum2: SpectrumParameters = SpectrumParameters {
        scale: 0.000,
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
        _LengthScale0: 15.0,
        _LengthScale1: 10.0,
        _LengthScale2: 8.0,
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
}

fn receive_displacement_texture(
    mut commands: Commands,
    displacement_receiver_query: Query<&DisplacementReceiver>,
    mut displacement_image_query: Query<&mut DisplacementImage>,
) {
    if let Ok(receiver) = displacement_receiver_query.get_single() {
        let result = receiver.receiver.try_recv();
        if let Ok(result) = result {
            if let Ok(mut d) = displacement_image_query.get_single_mut() {
                d.displacement = result.clone();
            } else {
                commands.spawn(DisplacementImage {
                    displacement: result.clone(),
                });
            }
            // commands.spawn(DisplacementImage {
            //     displacement: result.clone(),
            // });
            // let image = result.convert(TextureFormat::Rgba8UnormSrgb).unwrap();
            // let img = image.try_into_dynamic().unwrap();
            //
            // img.save("tmp.png").unwrap();
        }
    }
}

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

pub(crate) fn align_byte_size(value: u32) -> u32 {
    value + (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - (value % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
}

pub fn get_displacement(n: i32, scale: f32, image: &Image, pos: Vec2) -> Vec3 {
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

fn get_displacement_at_coord(scale: f32, image: &Image, pos: UVec2) -> Vec3 {
    let col = image.get_color_at(pos.x, pos.y).unwrap().to_linear();
    return Vec3::new(col.red * scale, col.green * scale, col.blue * scale);
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

pub fn get_adjusted_coords(n: i32, scale: f32, pos: Vec2, image: &Image) -> Vec2 {
    let mut adjustedPos: Vec2 = Vec2::from(pos);
    let mut factor = 1.0;
    for i in 0..3 {
        // let coords = get_coords(256, 202.0, adjustedPos, image);
        let displacement = get_displacement(n, 202.0, image, adjustedPos);
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
