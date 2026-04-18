use omnivoice_infer::postprocess::remove_silence;

#[test]
fn remove_silence_clamps_split_end_boundary_for_non_ms_aligned_input() {
    let input = vec![0.2; 23_999];

    let output = remove_silence(&input, 24_000, 200, 0, 0);

    assert_eq!(output.len(), input.len());
    assert!(output
        .iter()
        .zip(input.iter())
        .all(|(actual, expected)| (actual - expected).abs() < 5.0e-5));
}

#[test]
fn remove_silence_clamps_edge_trim_boundary_for_fully_silent_non_ms_aligned_input() {
    let output = remove_silence(&vec![0.0; 23_999], 24_000, 0, 0, 0);

    assert!(output.is_empty());
}
