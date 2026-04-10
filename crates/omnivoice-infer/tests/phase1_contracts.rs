mod support;

use std::path::Path;

use omnivoice_infer::{
    artifacts::{ArtifactCase, ReferenceArtifactBundle},
    contracts::{BatchedInputs, GenerationRequest, PreparedPrompt, VoiceClonePrompt},
    frontend::Frontend,
    postprocess::{cross_fade_chunks, fade_and_pad_audio, peak_normalize_auto_voice},
    stage0_loop::{build_timesteps, pack_cfg_batch, predict_tokens_with_scoring},
    stage0_model::Stage0WeightLayout,
    stage1_decoder::Stage1DecoderBundle,
};
use support::{model_root, reference_root};

#[test]
fn loads_debug_case_contract() {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case(ArtifactCase::DebugAutoEnShort).unwrap();
    let prepared = case.load_prepared_prompt().unwrap();

    assert_eq!(
        prepared.style_text,
        "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>"
    );
    assert_eq!(
        prepared.full_text,
        "OmniVoice creates clear speech from text with minimal setup."
    );
    assert_eq!(prepared.target_start_idx, 20);
    assert_eq!(prepared.prompt.input_ids_dims(), (1, 8, 114));
    assert_eq!(prepared.prompt.audio_mask_dims(), (1, 114));
}

#[test]
fn reproduces_debug_prompt_and_batch_contract() {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case(ArtifactCase::DebugAutoEnShort).unwrap();
    let reference = case.load_prepared_prompt().unwrap();

    let frontend = Frontend::from_model_root(model_root()).unwrap();
    let request = GenerationRequest::new_text_only(
        "OmniVoice creates clear speech from text with minimal setup.",
    )
    .with_language("en");
    let task = frontend.build_task(&request).unwrap();
    let prepared = frontend.prepare_prompt(&task, 0).unwrap();
    let batched = pack_cfg_batch(std::slice::from_ref(&prepared), task.target_lens()).unwrap();

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
    assert_eq!(
        prepared.prompt.input_ids_dims(),
        reference.prompt.input_ids_dims()
    );
    assert_eq!(
        prepared.prompt.audio_mask_dims(),
        reference.prompt.audio_mask_dims()
    );
    assert_eq!(prepared.target_start_idx, reference.target_start_idx);
    assert_eq!(batched.batch_input_ids_dims(), (2, 8, 114));
    assert_eq!(batched.batch_audio_mask_dims(), (2, 114));
    assert_eq!(batched.batch_attention_mask_dims(), (2, 1, 114, 114));
    assert_eq!(batched.tokens_init_dims(), (1, 8, 94));
}

#[test]
fn validates_debug_artifact_tensor_shapes() {
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case(ArtifactCase::DebugAutoEnShort).unwrap();
    let inputs = case.load_debug_inputs().unwrap();
    let forward = case.load_forward_step_zero().unwrap();
    let step_zero = case.load_step_capture(0).unwrap();

    assert_eq!(inputs.prepared_input_ids_dims(), (1, 8, 114));
    assert_eq!(inputs.batch_input_ids_dims(), (2, 8, 114));
    assert_eq!(forward.inputs_embeds_dims(), (2, 114, 1024));
    assert_eq!(forward.final_hidden_dims(), (2, 114, 1024));
    assert_eq!(step_zero.c_logits_dims(), (1, 8, 94, 1025));
    assert_eq!(step_zero.u_logits_dims(), (1, 8, 94, 1025));
    assert_eq!(step_zero.tokens_after_step_dims(), (1, 8, 94));
}

#[test]
fn stage0_namespace_accepts_only_expected_prefixes() {
    let layout = Stage0WeightLayout::from_model_root(model_root()).unwrap();

    assert!(layout.accepted_prefixes().contains(&"llm".to_string()));
    assert!(layout
        .accepted_prefixes()
        .contains(&"audio_embeddings".to_string()));
    assert!(layout
        .accepted_prefixes()
        .contains(&"audio_heads".to_string()));
    assert!(!layout
        .accepted_prefixes()
        .contains(&"audio_tokenizer".to_string()));
}

#[test]
fn stage1_loader_isolated_to_tokenizer_assets() {
    let bundle = Stage1DecoderBundle::from_model_root(model_root()).unwrap();

    assert!(bundle
        .weights_path()
        .ends_with(Path::new("audio_tokenizer").join("model.safetensors")));
    assert_eq!(bundle.output_sample_rate(), 24_000);
    assert_eq!(bundle.expected_codebooks(), 8);
}

#[test]
fn embedding_batch_contract_types_exist() {
    let prompt = PreparedPrompt::zeros((1, 8, 4), (1, 4), 2, 4);
    let batch = BatchedInputs::zeros((2, 8, 4), (2, 4), (2, 1, 4, 4), (1, 8, 2));
    let voice = VoiceClonePrompt::new_empty("hello.");

    assert_eq!(prompt.target_start_idx, 2);
    assert_eq!(batch.batch_input_ids_dims(), (2, 8, 4));
    assert_eq!(voice.ref_text, "hello.");
}

#[test]
fn iterative_loop_never_reintroduces_audio_mask_id() {
    let logits = case_logits();
    let pred = predict_tokens_with_scoring(&logits, &logits, 0.0, 0.0, 1024, 8, 1025).unwrap();
    assert!(pred.pred_tokens.iter().all(|token| *token != 1024));
}

#[test]
fn timestep_builder_matches_debug_schedule_length() {
    let steps = build_timesteps(0.0, 1.0, 33, 0.1).unwrap();
    assert_eq!(steps.len(), 34);
}

#[test]
fn postprocess_boundaries_behave() {
    let waveform = vec![0.25_f32; 24_000];
    let faded = fade_and_pad_audio(&waveform, 24_000, 0.1, 0.1);
    let normalized = peak_normalize_auto_voice(&waveform).unwrap();
    let mixed = cross_fade_chunks(&[waveform.clone(), waveform], 24_000, 0.05).unwrap();

    assert!(faded.len() > 24_000);
    assert!(
        normalized
            .iter()
            .fold(0.0_f32, |m: f32, v: &f32| m.max(v.abs()))
            <= 0.5 + 1e-6
    );
    assert!(mixed.len() > 24_000);
}

fn case_logits() -> Vec<f32> {
    let mut logits = vec![f32::NEG_INFINITY; 8 * 2 * 1025];
    for layer in 0..8 {
        let base = layer * 2 * 1025;
        logits[base + 3] = 0.0;
        logits[base + 1025 + 4] = 0.0;
        logits[base + 1024] = 10.0;
        logits[base + 1025 + 1024] = 10.0;
    }
    logits
}
