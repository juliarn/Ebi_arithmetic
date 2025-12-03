use crate::shader::matrix_mul::{
    GpuI128, GpuRationalU32, GpuRationalU64, GpuSignedU64, MatrixMulShader,
};
use std::sync::LazyLock;
use wgpu::{BufferSize, Features, ShaderModuleDescriptor};

pub static GPU_STATE: LazyLock<GpuState> = LazyLock::new(|| GpuState::new());
pub static COMPUTE_SHADERS: LazyLock<ComputeShaders> =
    LazyLock::new(|| ComputeShaders::new(&GPU_STATE));

pub enum GpuState {
    NotAvailable,
    Available {
        device: wgpu::Device,
        queue: wgpu::Queue,
    },
}

impl GpuState {
    fn new() -> Self {
        // Create an instance handling loading the actual graphic library
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        // Request GPU adapter
        let adapter_result =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));

        if !adapter_result.is_ok() {
            return GpuState::NotAvailable;
        }

        let adapter = adapter_result.unwrap();

        // Check for compute shader support
        let capabilities = adapter.get_downlevel_capabilities();
        if !capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            return GpuState::NotAvailable;
        }

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: adapter.features() & !Features::all_experimental_mask(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_or_else(
            |err| {
                eprintln!("Failed to create GPU device: {}", err);
                GpuState::NotAvailable
            },
            |(device, queue)| GpuState::Available { device, queue },
        )
    }

    pub fn is_available(&self) -> bool {
        !matches!(self, GpuState::NotAvailable)
    }

    pub fn get_device(&self) -> &wgpu::Device {
        match self {
            GpuState::Available { device, .. } => device,
            GpuState::NotAvailable => panic!("GpuState::get_device is not available"),
        }
    }

    pub fn get_queue(&self) -> &wgpu::Queue {
        match self {
            GpuState::Available { queue, .. } => queue,
            GpuState::NotAvailable => panic!("GpuState::get_queue is not available"),
        }
    }

    pub fn create_bind_group_layout_entry(
        &self,
        binding: u32,
        buffer_binding_type: wgpu::BufferBindingType,
        min_binding_size: Option<BufferSize>,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: buffer_binding_type,
                min_binding_size,
                has_dynamic_offset: false,
            },
            count: None,
        }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: ShaderModuleDescriptor<'_>,
        entry_point: Option<&'static str>,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = self.get_device().create_shader_module(desc);

        let pipeline_layout =
            self.get_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        self.get_device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

pub struct ComputeShaders {
    matrix_mul_shader_f32: MatrixMulShader<f32>,
    matrix_mul_shader_f64: MatrixMulShader<f64>,
    matrix_mul_shader_signed_u64: MatrixMulShader<GpuSignedU64>,
    matrix_mul_shader_i128: MatrixMulShader<GpuI128>,
    matrix_mul_shader_exact_u32: MatrixMulShader<GpuRationalU32>,
    matrix_mul_shader_exact_u64: MatrixMulShader<GpuRationalU64>,
}

impl ComputeShaders {
    pub fn new(gpu_state: &'static GpuState) -> Self {
        if !gpu_state.is_available() {
            return ComputeShaders {
                matrix_mul_shader_f32: MatrixMulShader::NotAvailable,
                matrix_mul_shader_f64: MatrixMulShader::NotAvailable,
                matrix_mul_shader_signed_u64: MatrixMulShader::NotAvailable,
                matrix_mul_shader_i128: MatrixMulShader::NotAvailable,
                matrix_mul_shader_exact_u32: MatrixMulShader::NotAvailable,
                matrix_mul_shader_exact_u64: MatrixMulShader::NotAvailable,
            };
        }

        let device = gpu_state.get_device();
        ComputeShaders {
            matrix_mul_shader_f32: MatrixMulShader::new(
                gpu_state,
                wgpu::include_wgsl!("matrix_mul_f32.wgsl"),
            ),
            matrix_mul_shader_f64: if device.features().contains(wgpu::Features::SHADER_F64) {
                MatrixMulShader::new(gpu_state, wgpu::include_wgsl!("matrix_mul_f64.wgsl"))
            } else {
                MatrixMulShader::NotAvailable
            },
            matrix_mul_shader_signed_u64: if device
                .features()
                .contains(wgpu::Features::SHADER_INT64)
            {
                MatrixMulShader::new(gpu_state, wgpu::include_wgsl!("matrix_mul_signed_u64.wgsl"))
            } else {
                MatrixMulShader::NotAvailable
            },
            matrix_mul_shader_i128: MatrixMulShader::new(
                gpu_state,
                wgpu::include_wgsl!("matrix_mul_i128.wgsl"),
            ),
            matrix_mul_shader_exact_u32: MatrixMulShader::new(
                gpu_state,
                wgpu::include_wgsl!("matrix_mul_exact_u32.wgsl"),
            ),
            matrix_mul_shader_exact_u64: if device
                .features()
                .contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS)
            {
                MatrixMulShader::new(gpu_state, wgpu::include_wgsl!("matrix_mul_exact_u64.wgsl"))
            } else {
                MatrixMulShader::NotAvailable
            },
        }
    }

    pub fn get_matrix_mul_shader_f32(&self) -> &MatrixMulShader<f32> {
        &self.matrix_mul_shader_f32
    }
    pub fn get_matrix_mul_shader_f64(&self) -> &MatrixMulShader<f64> {
        &self.matrix_mul_shader_f64
    }
    pub fn get_matrix_mul_shader_signed_u64(&self) -> &MatrixMulShader<GpuSignedU64> {
        &self.matrix_mul_shader_signed_u64
    }
    pub fn get_matrix_mul_shader_i128(&self) -> &MatrixMulShader<GpuI128> {
        &self.matrix_mul_shader_i128
    }
    pub fn get_matrix_mul_shader_exact_u32(&self) -> &MatrixMulShader<GpuRationalU32> {
        &self.matrix_mul_shader_exact_u32
    }
    pub fn get_matrix_mul_shader_exact_u64(&self) -> &MatrixMulShader<GpuRationalU64> {
        &self.matrix_mul_shader_exact_u64
    }
}
