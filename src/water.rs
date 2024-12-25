use bevy::asset::Handle;
use bevy::image::Image;
use bevy::prelude::{
    App, Commands, DirectAssetAccessExt, FromWorld, IntoSystemConfigs, Plugin, Res, Resource, Vec2,
    World,
};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::texture_storage_2d;
use bevy::render::render_resource::{
    AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
    StorageTextureAccess, TextureFormat,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::texture::{FallbackImage, GpuImage};
use bevy::render::{render_graph, Render, RenderApp, RenderSet};
use std::borrow::Cow;

const SHADER_ASSET_PATH: &str = "shaders/fft_water.wgsl";

const WORKGROUP_SIZE: u32 = 8;

pub struct WaterPlugin;

#[derive(Debug, Clone, ShaderType)]
pub struct SpectrumParameters {
    pub scale: f32,
    pub angle: f32,
    pub spreadBlend: f32,
    pub swell: f32,
    pub alpha: f32,
    pub peakOmega: f32,
    pub gamma: f32,
    pub shortWavesFade: f32,
}

#[derive(Resource, Clone, ExtractResource, AsBindGroup)]
pub struct WaterResource {
    #[uniform(0)]
    pub _N: u32,
    #[uniform(1)]
    pub _seed: i32,
    #[uniform(2)]
    pub _LengthScale0: u32,
    #[uniform(3)]
    pub _LengthScale1: u32,
    #[uniform(4)]
    pub _LengthScale2: u32,
    #[uniform(5)]
    pub _LowCutoff: f32,
    #[uniform(6)]
    pub _HighCutoff: f32,
    #[uniform(7)]
    pub _Gravity: f32,
    #[uniform(8)]
    pub _RepeatTime: f32,
    #[uniform(9)]
    pub _FrameTime: f32,

    #[uniform(10)]
    pub _Lambda: Vec2,

    #[storage(11)]
    pub _Spectrums: Handle<ShaderStorageBuffer>,

    #[storage_texture(12, image_format = Rgba32Float, access = WriteOnly, dimension = "2d_array")]
    pub _SpectrumTextures: Handle<Image>,

    #[storage_texture(13, image_format = Rgba32Float, access = WriteOnly, dimension = "2d_array")]
    pub _InitialSpectrumTextures: Handle<Image>,

    #[storage_texture(14, image_format = Rgba32Float, access = WriteOnly, dimension = "2d_array")]
    pub _DisplacementTextures: Handle<Image>,

    #[uniform(15)]
    pub _Depth: f32,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterLabel;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<WaterResource>::default());
        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(WaterLabel, WaterNode::default());
        render_graph.add_node_edge(WaterLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<WaterPipeline>();
    }
}

#[derive(Resource)]
struct WaterBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<WaterPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_shader_storage_buffer: Res<RenderAssets<GpuShaderStorageBuffer>>,
    fallback_images: Res<FallbackImage>,
    water_resource: Res<WaterResource>,
    render_device: Res<RenderDevice>,
) {
    // let _N = water_resource._N;
    // let displacement = gpu_images.get(&water_resource.displacement).unwrap();
    // let bind_group = render_device.create_bind_group(
    //     None,
    //     &pipeline.bind_group_layout,
    //     &BindGroupEntries::sequential((&displacement)),
    // );

    let mut bindGroupParam = (gpu_images, fallback_images, gpu_shader_storage_buffer);
    match water_resource.as_bind_group(
        &pipeline.bind_group_layout,
        &render_device,
        &mut bindGroupParam,
    ) {
        Ok(bind_group) => {
            commands.insert_resource(WaterBindGroup(bind_group.bind_group));
        }
        Err(e) => {
            eprintln!("Failed to create bind group: {}", e);
        }
    }
}

#[derive(Resource)]
struct WaterPipeline {
    bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for WaterPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let bind_group_layout = WaterResource::bind_group_layout(render_device);
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("CS_InitializeSpectrum"),
            zero_initialize_workgroup_memory: false,
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("CS_InitializeSpectrum"),
            zero_initialize_workgroup_memory: false,
        });

        WaterPipeline {
            bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum WaterState {
    Loading,
    Init,
    Update,
}

struct WaterNode {
    state: WaterState,
}

impl Default for WaterNode {
    fn default() -> Self {
        Self {
            state: WaterState::Loading,
        }
    }
}

impl render_graph::Node for WaterNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<WaterPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            WaterState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = WaterState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            WaterState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = WaterState::Update;
                }
            }
            WaterState::Update => {
                self.state = WaterState::Update;
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if let Some(bind_group) = &world.get_resource::<WaterBindGroup>() {
            let pipeline_cache = world.resource::<PipelineCache>();
            let pipeline = world.resource::<WaterPipeline>();
            {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                match self.state {
                    WaterState::Loading => {}
                    WaterState::Init => {
                        let init_pipeline = pipeline_cache
                            .get_compute_pipeline(pipeline.init_pipeline)
                            .unwrap();
                        pass.set_bind_group(0, &bind_group.0, &[]);
                        pass.set_pipeline(init_pipeline);
                        pass.dispatch_workgroups(256 / WORKGROUP_SIZE, 256 / WORKGROUP_SIZE, 1);
                    }
                    WaterState::Update => {
                        // let update_pipeline = pipeline_cache
                        //     .get_compute_pipeline(pipeline.update_pipeline)
                        //     .unwrap();
                        // pass.set_bind_group(0, &bind_group.0, &[]);
                        // pass.set_pipeline(update_pipeline);
                        // pass.dispatch_workgroups(
                        //     SIZE.0 / WORKGROUP_SIZE,
                        //     SIZE.1 / WORKGROUP_SIZE,
                        //     1,
                        // );
                    }
                }
            } // First pass ends here
        }

        Ok(())
    }
}
