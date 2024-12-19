use std::f32::consts::TAU;
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::Skybox;
use bevy::image::CompressedImageFormats;
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::{AsBindGroup, ShaderRef, TextureViewDescriptor, TextureViewDimension};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, TouchControls};

const SHADER_ASSET_PATH: &str = "shaders/custom_material.wgsl";

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MaterialPlugin::<CustomMaterial>::default()))
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        // .add_systems(Update, asset_loaded)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Create and save a handle to the mesh.
    let plane: Handle<Mesh> = meshes.add(Mesh::from(Plane3d::default().mesh().size(1000.0,1000.0).subdivisions(1000)));
    let skybox_handle = asset_server.load("textures/Ryfjallet_cubemap_bc7.ktx2");

    commands.spawn((
        Mesh3d(plane),
        MeshMaterial3d(materials.add(CustomMaterial {
            color: LinearRgba::new(0.0,0.2,1.0,1.0),
            skybox_texture: skybox_handle.clone(),
            alpha_mode: AlphaMode::Opaque,
        })),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));


    // camera
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera::default(),
        Skybox{
            brightness: 1000.0,
            image: skybox_handle.clone(),
            ..default()
        }
    ));
}

fn generate_custom_mesh() -> Mesh{
    let mut coordinates = Vec::new();
    let mut indices = Vec::new();
    let width = 2000;
    let depth = 2000;
    let spacing = 0.4;
    for z in 0..depth {
        for x in 0..width {
                coordinates.push([
                    x as f32 * spacing - (width as f32/2.0) * spacing,
                    0.0,
                    z as f32 * spacing- (depth as f32/2.0) * spacing,
                ]);
        }
    }

    for x in 0..width-1 {
        for z in 0..depth-1 {
            indices.extend([
                z * width + x,
                (z+1) * (width) +x,
                z * width + x + 1,
            ]);

            indices.extend([
                z * width + x + 1,
                (z+1) * width  +x,
                (z+1) * width + x+1,
            ]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, coordinates)
        .with_inserted_indices(Indices::U32(indices))
}


// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    color: LinearRgba,
    #[sampler(1)]
    #[texture(2, dimension="cube")]
    skybox_texture: Handle<Image>,

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
