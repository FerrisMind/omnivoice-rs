mod support;

use candle_core::{DType, Device};
use omnivoice_infer::{
    artifacts::{ArtifactCase, ReferenceArtifactBundle},
    contracts::{BoolTensor2, BoolTensor4, GenerationRequest, I64Tensor3},
    frontend::Frontend,
    pipeline::Phase3Pipeline,
    runtime::{auto_device_resolution_order, DTypeSpec, DeviceSpec, RuntimeOptions},
};
use support::{model_root, reference_root};

#[test]
fn runtime_options_default_to_auto_gpu_first() {
    let options = RuntimeOptions::new(model_root());
    let device = options.resolve_device().unwrap();

    assert_eq!(options.device(), DeviceSpec::Auto);
    assert_eq!(options.dtype(), DTypeSpec::Auto);
    if device.is_cuda() || device.is_metal() {
        assert_eq!(
            options.resolve_dtype_for_runtime_device(&device),
            DType::F16
        );
    } else {
        assert_eq!(
            options.resolve_dtype_for_runtime_device(&device),
            DType::F32
        );
    }
}

#[test]
fn auto_device_resolution_order_is_gpu_first() {
    let order = auto_device_resolution_order();
    assert_eq!(order.last().copied(), Some(DeviceSpec::Cpu));
    #[cfg(all(feature = "cuda", feature = "metal", target_os = "macos"))]
    assert_eq!(
        order,
        &[DeviceSpec::Cuda(0), DeviceSpec::Metal, DeviceSpec::Cpu]
    );
    #[cfg(all(
        feature = "cuda",
        not(all(feature = "metal", target_os = "macos"))
    ))]
    assert_eq!(order, &[DeviceSpec::Cuda(0), DeviceSpec::Cpu]);
    #[cfg(all(
        not(feature = "cuda"),
        all(feature = "metal", target_os = "macos")
    ))]
    assert_eq!(order, &[DeviceSpec::Metal, DeviceSpec::Cpu]);
    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "metal", target_os = "macos"))
    ))]
    assert_eq!(order, &[DeviceSpec::Cpu]);
}

#[test]
fn tensor_contracts_convert_to_candle() {
    let ids = I64Tensor3::full((1, 2, 3), 7);
    let mut mask2 = BoolTensor2::zeros((1, 3));
    mask2.set(0, 1, true);
    let mut mask4 = BoolTensor4::zeros((1, 1, 3, 3));
    mask4.set(0, 0, 0, 0, true);

    let ids_tensor = ids.to_candle(&Device::Cpu).unwrap();
    let mask2_tensor = mask2.to_candle(&Device::Cpu).unwrap();
    let mask4_tensor = mask4.to_candle(&Device::Cpu).unwrap();

    assert_eq!(ids_tensor.dims(), &[1, 2, 3]);
    assert_eq!(ids_tensor.dtype(), DType::I64);
    assert_eq!(mask2_tensor.dims(), &[1, 3]);
    assert_eq!(mask2_tensor.dtype(), DType::U8);
    assert_eq!(mask4_tensor.dims(), &[1, 1, 3, 3]);
    assert_eq!(mask4_tensor.dtype(), DType::U8);
}

#[test]
fn reproduces_auto_prompt_contract() {
    assert_prompt_matches_reference(ArtifactCase::DebugAutoEnShort);
}

#[test]
fn reproduces_clone_prompt_contract() {
    assert_prompt_matches_reference(ArtifactCase::CloneUserRef);
}

#[test]
fn reproduces_design_en_prompt_contract() {
    assert_prompt_matches_reference(ArtifactCase::DesignEnBritish);
}

#[test]
fn reproduces_design_zh_prompt_contract() {
    assert_prompt_matches_reference(ArtifactCase::DesignZhControl);
}

#[test]
fn duration_override_is_applied() {
    let frontend = Frontend::from_model_root(model_root()).unwrap();
    let request =
        GenerationRequest::new_text_only("Phase three duration override.").with_duration(2.0);

    let task = frontend.build_task(&request).unwrap();

    assert_eq!(task.target_lens, vec![50]);
}

#[test]
fn denoise_token_only_present_for_clone_prompt() {
    let frontend = Frontend::from_model_root(model_root()).unwrap();

    let auto_request = GenerationRequest::new_text_only("Auto voice should not include denoise.");
    let auto_task = frontend.build_task(&auto_request).unwrap();
    let auto_prepared = frontend.prepare_prompt(&auto_task, 0).unwrap();
    assert!(!auto_prepared.style_text.contains("<|denoise|>"));

    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let clone_case = bundle.case(ArtifactCase::CloneUserRef).unwrap();
    let clone_request = clone_case.build_generation_request().unwrap();
    let clone_task = frontend.build_task(&clone_request).unwrap();
    let clone_prepared = frontend.prepare_prompt(&clone_task, 0).unwrap();
    assert!(clone_prepared.style_text.contains("<|denoise|>"));
}

#[test]
fn stage1_prepare_loads_reference_tokens_into_candle() {
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();

    let decode = pipeline
        .prepare_stage1_from_reference_case(reference_root(), "debug_auto_en_short")
        .unwrap();

    assert_eq!(decode.token_dims().unwrap(), (1, 8, 94));
    assert_eq!(decode.tokens.dtype(), DType::I64);
    assert_eq!(decode.sample_rate, 24_000);
    assert_eq!(decode.hop_length, 960);
    assert_eq!(decode.expected_codebooks, 8);
}

#[test]
fn prepare_prompt_builds_candle_backed_batch() {
    let pipeline = Phase3Pipeline::from_options(RuntimeOptions::new(model_root())).unwrap();

    let batch = pipeline
        .prepare_prompt_from_reference_case(reference_root(), "debug_auto_en_short")
        .unwrap();

    assert_eq!(batch.input_ids_dims().unwrap(), (2, 8, 114));
    assert_eq!(batch.audio_mask_dims().unwrap(), (2, 114));
    assert_eq!(batch.attention_mask_dims().unwrap(), (2, 1, 114, 114));
    assert_eq!(batch.tokens_init_dims().unwrap(), (1, 8, 94));
    assert_eq!(batch.input_ids.dtype(), DType::I64);
}

fn assert_prompt_matches_reference(case: ArtifactCase) {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case(case).unwrap();
    let reference = case.load_prepared_prompt().unwrap();
    let request = case.build_generation_request().unwrap();

    let frontend = Frontend::from_model_root(model_root()).unwrap();
    let task = frontend.build_task(&request).unwrap();
    let prepared = frontend.prepare_prompt(&task, 0).unwrap();

    assert_eq!(prepared.mode, reference.mode);
    assert_eq!(prepared.style_text, reference.style_text);
    assert_eq!(prepared.full_text, reference.full_text);
    assert_eq!(prepared.style_token_ids, reference.style_token_ids);
    assert_eq!(prepared.text_token_ids, reference.text_token_ids);
    assert_eq!(
        prepared.prompt.input_ids.data,
        reference.prompt.input_ids.data
    );
    assert_eq!(
        prepared.prompt.audio_mask.data,
        reference.prompt.audio_mask.data
    );
    assert_eq!(prepared.target_start_idx, reference.target_start_idx);
    assert_eq!(prepared.total_length, reference.total_length);
}
