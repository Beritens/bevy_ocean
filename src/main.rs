mod water;
mod water_compute;

use crate::water::{get_adjusted_coords, get_displacement, get_normal, DisplacementImage, SkyCubeMap, SlopeImage, WaterPlugin};
use avian3d::parry::na::clamp;
use avian3d::prelude::{AngularDamping, AngularVelocity, CenterOfMass, Collider, ColliderDensity, ComputedMass, ExternalForce, Gravity, LinearDamping, LinearVelocity, Mass, Position, RigidBody};
use avian3d::PhysicsPlugins;
use bevy::asset::saver::ErasedAssetSaver;
use bevy::core_pipeline::Skybox;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSamplerDescriptor};
use bevy::prelude::ops::powf;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, TextureViewDimension};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use wgpu::TextureViewDescriptor;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins.set(ImagePlugin {
            default_sampler: ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                address_mode_w: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                ..Default::default()
            },
        }),))
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(WaterPlugin)
        .add_plugins(PhysicsPlugins::default())
        .add_systems(Startup, setup)
        .add_systems(FixedPreUpdate, (movement))
        .add_systems(Update, (debug_swim, update_skybox))
        .add_systems(FixedUpdate, bad_swim)
        .add_systems(FixedUpdate, swim)
        .insert_resource(Gravity(Vec3::Y * -9.81))
        .run();
}

#[derive(Component)]
struct Buoyant {
    voxels: UVec3,
    size: Vec3,
    offset: Vec3,
}

#[derive(Component)]
struct Debug {
}

#[derive(Component)]
struct Board {
    speed: f32
}

#[derive(Component)]
struct Control {}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let skybox_handle = asset_server.load("textures/Ryfjallet_cubemap_bc7.ktx2");

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

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-2.6),
            ..default()
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
    ));

    // for x in -20..20 {
    //     for z in -20..20 {
    //         commands.spawn((
    //             Buoyant {
    //                 voxels: UVec3::new(3, 3, 3),
    //                 size: Vec3::new(1.0, 1.0, 1.0),
    //                 offset: Vec3::new(0.0, 0.0, 0.0),
    //             },
    //             Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
    //             MeshMaterial3d(materials.add(Color::WHITE)),
    //             Transform::from_xyz(2.0 * x as f32, 5.0, 2.0 * z as f32)
    //                 .with_scale(Vec3::splat(1.0)),
    //             RigidBody::Dynamic,
    //             Collider::cuboid(1.0, 1.0, 1.0),
    //             CenterOfMass::new(0.0, 0.0, 0.0),
    //             ColliderDensity(0.2),
    //             LinearDamping(0.01),
    //             AngularDamping(0.1),
    //         ));
    //     }
    // }
    // let cube_mesh = meshes.add(Cuboid::default());
    // commands.spawn((
    //     Mesh3d(cube_mesh.clone()),
    //     Debug{},
    //     // Board{speed: 0.0},
    //     // Buoyant {
    //     //     voxels: UVec3::new(10, 1, 10),
    //     //     size: Vec3::new(1.0, 1.0, 1.0),
    //     //     offset: Vec3::new(0.0, 0.0, 0.0),
    //     // },
    //     MeshMaterial3d(materials.add(Color::srgb(2.0, 2.0, 2.0))),
    //     Transform::from_translation(Vec3::new(0.0, 1.0, 0.0))
    //         .with_scale(Vec3::new(1.2, 0.3, 3.0)),
    //     // RigidBody::Dynamic,
    //     // Collider::cuboid(1.0, 1.0, 1.0),
    //     // CenterOfMass::new(0.0, 0.0, 0.0),
    //     // ColliderDensity(0.2),
    //     // Control {},
    // ));
}

fn movement(
    keys: Res<ButtonInput<KeyCode>>,
    mut characters: Query<(&GlobalTransform, &mut Board, &mut LinearVelocity, &mut AngularVelocity), With<Control>>,
) {
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
    for (transform, mut board, mut vel, mut ang) in characters.iter_mut() {
        board.speed += input_vector.y;
        vel.0 = transform.forward() * board.speed;
        ang.0 = - 2.0 * transform.up() * input_vector.x;
        // vel.0 += transform.right() * input_vector.x;
        // transform.translation.x += 2.0 * input_vector.x;
        // transform.translation.z -= 2.0 * input_vector.y;
    }
}

fn calculate_displaced_amount(pos: Vec3, displacement: Vec3, box_size: Vec3) -> f32 {
    let displaced_height = clamp(displacement.y - (pos.y - box_size.y / 2.0), 0.0, box_size.y);
    displaced_height * box_size.x * box_size.z
}

fn calculate_buoyancy(displaced_amount: f32, normal: Vec3) -> Vec3 {
    9.81 * normal * (displaced_amount)
}

fn calculate_drag(v: Vec3, displaced_amount: f32, water_drag: f32) -> Vec3 {
    if (displaced_amount <= 0.0) {
        return Vec3::ZERO;
    }
    return -v * v.length() * water_drag;
}

// fn swim_board(
//     mut transform_query: Query<(&mut Transform), With<Board>>,
//     displacement_image_query: Query<(&DisplacementImage)>,
// ) {
// }

fn debug_swim(
    mut buoyant_query: Query<(&mut Transform), (With<Debug>)>,
    displacement_image_query: Query<(&DisplacementImage)>,
    slope_image_query: Query<(&SlopeImage)>,
) {
    if let (Ok(displacement_image), Ok(slope_image)) = (displacement_image_query.get_single(), slope_image_query.get_single() ) {
        let image = &displacement_image.displacement;
        let scale = 102.0;
        let n = 256;
        buoyant_query.par_iter_mut().for_each(|(mut transform)| {
            let coords = get_adjusted_coords(
                n,
                scale,
                Vec2::new(transform.translation.x, transform.translation.z),
                image,
            );
            let displacement =
                get_displacement(n, scale, &displacement_image.displacement, coords);

            let normal =
                get_normal(n, scale, &slope_image.slope, coords);
            transform.translation.y = displacement.y;
            transform.look_to(normal, Vec3::X);
        });
    }
}
fn bad_swim(
    mut buoyant_query: Query<(&mut Transform, &mut LinearVelocity), (With<Board>)>,
    displacement_image_query: Query<(&DisplacementImage)>,
    slope_image_query: Query<(&SlopeImage)>,
) {
    if let (Ok(displacement_image), Ok(slope_image)) = (displacement_image_query.get_single(), slope_image_query.get_single() ) {
        let image = &displacement_image.displacement;
        let scale = 102.0;
        let n = 256;
        buoyant_query.par_iter_mut().for_each(|(mut transform, mut linear_velocity)| {
            let coords = get_adjusted_coords(
                n,
                scale,
                Vec2::new(transform.translation.x, transform.translation.z),
                image,
            );
            let displacement =
                get_displacement(n, scale, &displacement_image.displacement, coords);

            let normal =
                get_normal(n, scale, &slope_image.slope, coords);
            transform.translation.y = displacement.y;
            linear_velocity.y = 0.0;
        });
    }
}

fn swim(
    time: Res<Time>,
    mut buoyant_query: Query<
        (
            &mut ExternalForce,
            &GlobalTransform,
            &ComputedMass,
            &Buoyant,
            &CenterOfMass,
            &mut LinearVelocity,
            &AngularVelocity,
        ),
        With<Buoyant>,
    >,
    displacement_image_query: Query<(&DisplacementImage)>,
    slope_image_query: Query<(&SlopeImage)>,
) {
    if let (Ok(displacement_image), Ok(slope_image)) = (displacement_image_query.get_single(), slope_image_query.get_single() ) {
        let d_image = &displacement_image.displacement;
        let s_image = &slope_image.slope;
        let scale = 102.0;
        let n = 256;
        buoyant_query.par_iter_mut().for_each(
            |(
                mut external_force,
                transform,
                mass,
                buoyant,
                center_of_mass,
                mut linear_velocity,
                angular_veloctiy,
            )| {
                external_force.persistent = false;
                let voxelSize = Vec3::new(
                    buoyant.size.x / buoyant.voxels.x as f32,
                    buoyant.size.y / buoyant.voxels.y as f32,
                    buoyant.size.z / buoyant.voxels.z as f32,
                );
                let pos = transform.translation();
                let transform_scale = transform.scale();
                let com = transform.transform_point(center_of_mass.0 / transform_scale);
                for x in 0..buoyant.voxels.x {
                    for y in 0..buoyant.voxels.y {
                        for z in 0..buoyant.voxels.z {
                            //this could be precomputed
                            let mut local_point = Vec3::new(
                                x as f32 * (buoyant.size.x / buoyant.voxels.x as f32)
                                    - (buoyant.size.x / 2.0 - voxelSize.x / 2.0),
                                y as f32 * (buoyant.size.y / buoyant.voxels.y as f32)
                                    - (buoyant.size.y / 2.0 - voxelSize.y / 2.0),
                                z as f32 * (buoyant.size.z / buoyant.voxels.z as f32)
                                    - (buoyant.size.z / 2.0 - voxelSize.z / 2.0),
                            );
                            local_point += buoyant.offset;
                            let global_point = transform.transform_point(local_point);

                            let coords = get_adjusted_coords(
                                n,
                                scale,
                                Vec2::new(global_point.x, global_point.z),
                                d_image,
                            );
                            let displacement = get_displacement(
                                n,
                                scale,
                                &d_image,
                                coords,
                            );

                            let normal = get_normal(
                                n,
                                scale,
                                &s_image,
                                coords,
                            );

                            let displaced_amount = calculate_displaced_amount(
                                global_point,
                                displacement,
                                voxelSize * transform_scale,
                            );
                            let velocity = 1.0 * linear_velocity.0
                                + 1.0 * angular_veloctiy.cross(global_point - com);

                            let drag = calculate_drag(
                                velocity,
                                displaced_amount,
                                0.015
                                    * (voxelSize.x
                                        * transform_scale.x
                                        * voxelSize.z
                                        * transform_scale.z),
                            );
                            let b_foce = calculate_buoyancy(displaced_amount, normal);
                            external_force.apply_force_at_point(b_foce, global_point, com);
                            external_force.apply_force_at_point(drag, global_point, com);
                        }
                    }
                }
            },
        );
    }
}

pub fn update_skybox(
    mut cubemap: ResMut<SkyCubeMap>,
    mut skybox_query: Query<&mut Skybox>
) {
    if !cubemap.loaded {
        if let Ok(mut skybox) = skybox_query.get_single_mut(){
            skybox.image = cubemap.image.clone();
        }
    }
}
