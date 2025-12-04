use crate::shader::state::GpuState;
use std::marker::PhantomData;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRationalU64 {
    pub num: u64,
    pub den: u64,
    pub sign: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRationalU32 {
    pub num: u32,
    pub den: u32,
    pub sign: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSignedU64 {
    pub value: u64,
    pub sign: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuI128 {
    pub number: [u32; 4],
    pub sign: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Dimensions {
    pub n: u32,
    pub m: u32,
    pub p: u32,
}

pub enum MatrixMulShader<T: bytemuck::Pod> {
    NotAvailable,
    Available {
        gpu_state: &'static GpuState,
        bind_group_layout: wgpu::BindGroupLayout,
        compute_pipeline: wgpu::ComputePipeline,
        _marker: PhantomData<T>,
    },
}

impl<T: bytemuck::Pod> MatrixMulShader<T>
where
    T: bytemuck::Pod,
{
    pub fn new(gpu_state: &'static GpuState, desc: wgpu::ShaderModuleDescriptor) -> Self {
        if !gpu_state.is_available() {
            return MatrixMulShader::NotAvailable;
        }

        let bind_group_layout =
            gpu_state
                .get_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        // A Buffer
                        gpu_state.create_bind_group_layout_entry(
                            0,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            None,
                        ),
                        // B Buffer
                        gpu_state.create_bind_group_layout_entry(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            None,
                        ),
                        // Dimensions Uniform,
                        gpu_state.create_bind_group_layout_entry(
                            2,
                            wgpu::BufferBindingType::Uniform,
                            None,
                        ),
                        // C Buffer
                        gpu_state.create_bind_group_layout_entry(
                            3,
                            wgpu::BufferBindingType::Storage { read_only: false },
                            None,
                        ),
                    ],
                });

        let compute_pipeline =
            gpu_state.create_compute_pipeline(desc, Some("mul"), &bind_group_layout);

        MatrixMulShader::Available {
            gpu_state,
            bind_group_layout,
            compute_pipeline,
            _marker: PhantomData,
        }
    }

    pub fn is_available(&self) -> bool {
        !matches!(self, MatrixMulShader::NotAvailable)
    }

    pub fn execute(&self, a: Vec<T>, b: Vec<T>, dims: Dimensions) -> Vec<T> {
        match self {
            MatrixMulShader::NotAvailable => panic!("MatrixMulShader not available"),
            MatrixMulShader::Available {
                gpu_state,
                bind_group_layout,
                compute_pipeline,
                _marker,
            } => {
                let device = gpu_state.get_device();

                // Input matrix A
                let a_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Matrix A"),
                    contents: bytemuck::cast_slice(a.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                // Input matrix B
                let b_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Matrix B"),
                    contents: bytemuck::cast_slice(b.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                // Uniform buffer for dimensions
                let dims_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dimensions"),
                    contents: bytemuck::bytes_of(&dims),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                // Output buffer C
                let c_buffer_size =
                    (dims.n * dims.p * size_of::<T>() as u32) as wgpu::BufferAddress;
                let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Matrix C"),
                    size: c_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // Download buffer for C
                let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Download Matrix C"),
                    size: c_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: c_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        dims.n.div_ceil(16u32),
                        dims.p.div_ceil(16u32),
                        1,
                    );
                    drop(compute_pass);

                    encoder.copy_buffer_to_buffer(&c_buffer, 0, &download_buffer, 0, c_buffer_size);
                }

                let command_buffer = encoder.finish();
                gpu_state.get_queue().submit([command_buffer]);

                let buffer_slice = download_buffer.slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                let data = buffer_slice.get_mapped_range();
                let result: &[T] = bytemuck::cast_slice(&data);

                result.to_vec()
            }
        }
    }
}
