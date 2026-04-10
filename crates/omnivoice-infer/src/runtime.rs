use std::path::{Path, PathBuf};

use candle_core::{DType, Device};

use crate::{
    artifacts::RuntimeArtifacts,
    error::{OmniVoiceError, Result},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSpec {
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    Metal,
}

impl DeviceSpec {
    pub fn parse(value: &str) -> Result<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized == "auto" {
            return Ok(Self::Auto);
        }
        if normalized == "cpu" {
            return Ok(Self::Cpu);
        }
        if normalized == "metal" {
            return Ok(Self::Metal);
        }
        if let Some(index) = normalized.strip_prefix("cuda:") {
            let ordinal = index.parse::<usize>().map_err(|_| {
                OmniVoiceError::InvalidRequest(format!("invalid cuda device ordinal in {value}"))
            })?;
            return Ok(Self::Cuda(ordinal));
        }
        Err(OmniVoiceError::InvalidRequest(format!(
            "unsupported device spec {value}"
        )))
    }

    pub fn resolve(self) -> Result<Device> {
        match self {
            Self::Auto => resolve_auto_device(),
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(index) => resolve_cuda_device(index),
            Self::Metal => resolve_metal_device(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DTypeSpec {
    #[default]
    Auto,
    F32,
    F16,
    BF16,
}

impl DTypeSpec {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            other => Err(OmniVoiceError::InvalidRequest(format!(
                "unsupported dtype spec {other}"
            ))),
        }
    }

    pub fn resolve_for_device(self, device: DeviceSpec) -> DType {
        match self {
            Self::Auto => match device {
                DeviceSpec::Cuda(_) | DeviceSpec::Metal => DType::F16,
                DeviceSpec::Auto | DeviceSpec::Cpu => DType::F32,
            },
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }

    pub fn resolve_for_runtime_device(self, device: &Device) -> DType {
        match self {
            Self::Auto if device.is_cuda() || device.is_metal() => DType::F16,
            Self::Auto => DType::F32,
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOptions {
    model_root: PathBuf,
    device: DeviceSpec,
    dtype: DTypeSpec,
    seed: Option<u64>,
}

impl RuntimeOptions {
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            device: DeviceSpec::default(),
            dtype: DTypeSpec::default(),
            seed: None,
        }
    }

    pub fn with_device(mut self, device: DeviceSpec) -> Self {
        self.device = device;
        self
    }

    pub fn with_dtype(mut self, dtype: DTypeSpec) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn device(&self) -> DeviceSpec {
        self.device
    }

    pub fn dtype(&self) -> DTypeSpec {
        self.dtype
    }

    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    pub fn resolve_device(&self) -> Result<Device> {
        self.device.resolve()
    }

    pub fn resolve_dtype(&self) -> DType {
        self.dtype.resolve_for_device(self.device)
    }

    pub fn resolve_dtype_for_runtime_device(&self, device: &Device) -> DType {
        self.dtype.resolve_for_runtime_device(device)
    }

    pub fn load_runtime_artifacts(&self) -> Result<RuntimeArtifacts> {
        RuntimeArtifacts::from_model_root(&self.model_root)
    }
}

#[cfg(all(feature = "cuda", feature = "metal", target_os = "macos"))]
const AUTO_DEVICE_ORDER: [DeviceSpec; 3] =
    [DeviceSpec::Cuda(0), DeviceSpec::Metal, DeviceSpec::Cpu];

#[cfg(all(
    feature = "cuda",
    not(all(feature = "metal", target_os = "macos"))
))]
const AUTO_DEVICE_ORDER: [DeviceSpec; 2] = [DeviceSpec::Cuda(0), DeviceSpec::Cpu];

#[cfg(all(
    not(feature = "cuda"),
    all(feature = "metal", target_os = "macos")
))]
const AUTO_DEVICE_ORDER: [DeviceSpec; 2] = [DeviceSpec::Metal, DeviceSpec::Cpu];

#[cfg(all(
    not(feature = "cuda"),
    not(all(feature = "metal", target_os = "macos"))
))]
const AUTO_DEVICE_ORDER: [DeviceSpec; 1] = [DeviceSpec::Cpu];

pub fn auto_device_resolution_order() -> &'static [DeviceSpec] {
    &AUTO_DEVICE_ORDER
}

fn resolve_auto_device() -> Result<Device> {
    for candidate in auto_device_resolution_order() {
        match candidate {
            DeviceSpec::Cuda(index) => {
                #[cfg(feature = "cuda")]
                if let Ok(device) = Device::new_cuda(*index) {
                    return Ok(device);
                }
                #[cfg(not(feature = "cuda"))]
                let _ = index;
            }
            DeviceSpec::Metal =>
            {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if let Ok(device) = Device::new_metal(0) {
                    return Ok(device);
                }
            }
            DeviceSpec::Cpu => return Ok(Device::Cpu),
            DeviceSpec::Auto => {}
        }
    }
    Ok(Device::Cpu)
}

#[cfg(feature = "cuda")]
fn resolve_cuda_device(index: usize) -> Result<Device> {
    Ok(Device::new_cuda(index)?)
}

#[cfg(not(feature = "cuda"))]
fn resolve_cuda_device(index: usize) -> Result<Device> {
    Err(OmniVoiceError::Unsupported(format!(
        "cuda device cuda:{index} requires the `cuda` feature"
    )))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resolve_metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn resolve_metal_device() -> Result<Device> {
    Err(OmniVoiceError::Unsupported(
        "metal device requires the `metal` feature".to_string(),
    ))
}
