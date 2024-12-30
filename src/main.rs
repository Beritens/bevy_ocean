mod water;
mod water_compute;

use crate::water::{get_adjusted_coords, get_displacement, DisplacementImage, WaterPlugin};
use avian3d::parry::na::clamp;
use avian3d::prelude::{
    AngularVelocity, CenterOfMass, Collider, ColliderDensity, ComputedMass, ExternalForce, Gravity,
    LinearVelocity, Mass, Position, RigidBody,
};
use avian3d::PhysicsPlugins;
use bevy::asset::saver::ErasedAssetSaver;
use bevy::core_pipeline::Skybox;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

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

    // for x in -30..30 {
    //     for z in -30..30 {
    //         commands.spawn((
    //             Buoyant {},
    //             Control {},
    //             Mesh3d(meshes.add(Cuboid::new(5.0, 5.0, 5.0))),
    //             MeshMaterial3d(materials.add(Color::WHITE)),
    //             Transform::from_xyz(10.0 * x as f32, 50.0, 10.0 * z as f32),
    //         ));
    //     }
    // }
    let cube_mesh = meshes.add(Cuboid::default());
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        Buoyant {
            voxels: UVec3::new(10, 1, 10),
            size: Vec3::new(1.0, 1.0, 1.0),
            offset: Vec3::new(0.0, 0.0, 0.0),
        },
        MeshMaterial3d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        Transform::from_translation(Vec3::new(0.0, -100.0, 0.0))
            .with_scale(Vec3::new(10.0, 1.0, 10.0)),
        RigidBody::Dynamic,
        Collider::cuboid(1.0, 1.0, 1.0),
        CenterOfMass::new(0.0, 0.0, 0.0),
        ColliderDensity(0.2),
        Control {},
    ));
}

fn movement(
    keys: Res<ButtonInput<KeyCode>>,
    mut characters: Query<(&GlobalTransform, &mut LinearVelocity), With<Control>>,
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
    for (transform, mut vel) in characters.iter_mut() {
        vel.0 += transform.forward() * input_vector.y;
        vel.0 += transform.right() * input_vector.x;
        // transform.translation.x += 2.0 * input_vector.x;
        // transform.translation.z -= 2.0 * input_vector.y;
    }
}

fn calculate_displaced_amount(pos: Vec3, displacement: Vec3, box_size: Vec3) -> f32 {
    let displaced_height = clamp(displacement.y - (pos.y - box_size.y / 2.0), 0.0, box_size.y);
    displaced_height * box_size.x * box_size.z
}

fn calculate_buoyancy(displaced_amount: f32) -> Vec3 {
    9.81 * Vec3::Y * (displaced_amount)
}

fn calculate_drag(v: Vec3, displaced_amount: f32, water_drag: f32) -> Vec3 {
    if (displaced_amount <= 0.0) {
        return Vec3::ZERO;
    }
    return -v * v.length() * displaced_amount * water_drag;
}

fn swim(
    mut buoyant_query: Query<
        (
            &mut ExternalForce,
            &GlobalTransform,
            &ComputedMass,
            &Buoyant,
            &CenterOfMass,
            &LinearVelocity,
            &AngularVelocity,
        ),
        With<Buoyant>,
    >,
    displacement_image_query: Query<(&DisplacementImage)>,
) {
    if let Ok(displacement_image) = displacement_image_query.get_single() {
        let image = &displacement_image.displacement;
        let scale = 202.0;
        let n = 256;
        buoyant_query.par_iter_mut().for_each(
            |(
                mut external_force,
                transform,
                mass,
                buoyant,
                center_of_mass,
                linear_velocity,
                angular_veloctiy,
            )| {
                external_force.persistent = false;
                let voxelSize = Vec3::new(
                    buoyant.size.x / buoyant.voxels.x as f32,
                    buoyant.size.y / buoyant.voxels.y as f32,
                    buoyant.size.z / buoyant.voxels.z as f32,
                );
                println!("{}", center_of_mass.0);
                let com = transform.transform_point(center_of_mass.0 / transform.scale());
                println!("{}", com);
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
                                image,
                            );
                            let displacement = get_displacement(
                                256,
                                202.0,
                                &displacement_image.displacement,
                                coords,
                            );

                            let displaced_amount = calculate_displaced_amount(
                                global_point,
                                displacement,
                                voxelSize * transform.scale(),
                            );
                            let velocity = 1.0 * linear_velocity.0
                                + 1.0 * angular_veloctiy.cross(global_point - com);

                            let drag = calculate_drag(
                                velocity,
                                displaced_amount,
                                0.2 / (voxelSize.x
                                    * transform.scale().x
                                    * voxelSize.z
                                    * transform.scale().z),
                            );
                            let b_foce = calculate_buoyancy(displaced_amount);
                            external_force.apply_force_at_point(b_foce, global_point, com);
                            external_force.apply_force_at_point(drag, global_point, com);
                        }
                    }
                }

                // position.y = displacement.y;
            },
        );
    }
}
