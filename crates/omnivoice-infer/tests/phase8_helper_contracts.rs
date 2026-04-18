use omnivoice_infer::{
    audio_input::{root_mean_square, ReferenceAudioProcessor},
    contracts::{ReferenceAudioInput, WaveformInput},
    frontend::combine_text,
    postprocess::{
        apply_clone_rms_restore, fade_and_pad_audio, peak_normalize_auto_voice, remove_silence,
    },
};

#[test]
fn combine_text_keeps_laughter_but_strips_space_before_other_emotion_tags() {
    assert_eq!(combine_text("hello [laughter]", None), "hello [laughter]");
    assert_eq!(
        combine_text("hello [surprise-ah]", None),
        "hello[surprise-ah]"
    );
}

#[test]
fn prepare_prompt_audio_applies_rms_boost_and_hop_truncation_without_punctuation() {
    let processor = ReferenceAudioProcessor::new(1_000, 10);
    let input = ReferenceAudioInput::Waveform(WaveformInput::mono(vec![0.05; 23], 1_000));

    let prepared = processor
        .prepare_prompt_audio(&input, Some("hello"), false)
        .unwrap();

    assert_eq!(prepared.ref_text.as_deref(), Some("hello"));
    assert_eq!(prepared.waveform.len(), 20);
    assert_eq!(prepared.waveform.len() % 10, 0);
    assert!(matches!(prepared.ref_rms, Some(rms) if (rms - 0.05).abs() < 1.0e-6));
    assert!(prepared
        .waveform
        .iter()
        .all(|sample| (*sample - 0.1).abs() < 1.0e-6));
}

#[test]
fn prepare_prompt_audio_preprocess_skips_long_trim_when_ref_text_is_provided() {
    let processor = ReferenceAudioProcessor::new(1_000, 10);
    let mut samples = vec![0.2; 10_000];
    samples.extend(vec![0.0; 1_000]);
    samples.extend(vec![0.2; 10_000]);
    let input = ReferenceAudioInput::Waveform(WaveformInput::mono(samples, 1_000));

    let with_ref_text = processor
        .prepare_prompt_audio(&input, Some("hello"), true)
        .unwrap();
    let without_ref_text = processor.prepare_prompt_audio(&input, None, true).unwrap();

    assert_eq!(with_ref_text.ref_text.as_deref(), Some("hello."));
    assert_eq!(with_ref_text.waveform.len() % 10, 0);
    assert!(with_ref_text.waveform.len() > 18_000);
    assert!(without_ref_text.waveform.len() < 12_000);
    assert!(with_ref_text.waveform.len() > without_ref_text.waveform.len() + 7_000);
}

#[test]
fn prepare_prompt_audio_preprocess_clamps_non_ms_aligned_reference_audio() {
    let processor = ReferenceAudioProcessor::new(24_000, 10);
    let input = ReferenceAudioInput::Waveform(WaveformInput::mono(vec![0.2; 23_999], 24_000));

    let prepared = processor.prepare_prompt_audio(&input, None, true).unwrap();

    assert_eq!(prepared.ref_text, None);
    assert_eq!(prepared.waveform.len(), 23_990);
    assert_eq!(prepared.waveform.len() % 10, 0);
}

#[test]
fn postprocess_helpers_match_python_volume_and_padding_contracts() {
    let auto_normalized = peak_normalize_auto_voice(&[1.0, -0.25]).unwrap();
    let auto_peak = auto_normalized
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!((auto_peak - 0.5).abs() < 1.0e-6);

    let clone_restored = apply_clone_rms_restore(&[0.2, -0.2], 0.05);
    assert_eq!(clone_restored, vec![0.1, -0.1]);

    let faded = fade_and_pad_audio(&vec![0.25; 1_000], 1_000, 0.1, 0.1);
    assert_eq!(faded.len(), 1_200);
    assert!(faded[..100].iter().all(|sample| *sample == 0.0));
    assert!(faded[faded.len() - 100..]
        .iter()
        .all(|sample| *sample == 0.0));

    let mut segmented = vec![0.0; 50];
    segmented.extend(vec![0.2; 100]);
    segmented.extend(vec![0.0; 50]);
    segmented.extend(vec![0.2; 100]);
    segmented.extend(vec![0.0; 50]);
    let compact = remove_silence(&segmented, 1_000, 20, 10, 10);
    assert!(compact.len() < segmented.len());
    assert!(root_mean_square(&compact).unwrap() > 0.05);
}
