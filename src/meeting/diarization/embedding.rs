//! Local speaker-embedding diarization using ONNX Runtime.
//!
//! This backend keeps the current low-risk meeting behavior for microphone
//! audio (`You`) and applies online speaker matching to loopback segments.
//! It is designed around WeSpeaker-style ONNX embedding models.

use super::{DiarizationConfig, DiarizedSegment, Diarizer, SpeakerId};
use crate::config::Config;
use crate::meeting::data::AudioSource;
use crate::meeting::TranscriptSegment;
use std::collections::HashMap;
use std::path::PathBuf;

const DEFAULT_MODEL_NAME: &str = "wespeaker-resnet221-lm";
const SAMPLE_RATE: u64 = 16_000;
const EMA_ALPHA: f32 = 0.05;
const SPEAKER_WINDOW_HOP_MIN_MS: u64 = 250;
const SPEAKER_WINDOW_HOP_MAX_MS: u64 = 500;
const NEW_SPEAKER_MIN_WINDOWS: usize = 2;
const NEW_SPEAKER_MIN_RUN_MS: u64 = 1_500;
const MIN_DOMINANT_SPEAKER_SHARE: f32 = 0.60;
const AMBIGUOUS_COMPETITOR_SHARE: f32 = 0.20;
const NEW_SPEAKER_CONFIDENCE: f32 = 0.80;
const TRANSIENT_SPEAKER_RUN_MS: u64 = 800;
const MIN_SPLIT_RUN_MS: u64 = 1_200;
const MIN_SPLIT_RUN_SHARE: f32 = 0.18;
const MAX_SPEAKER_SPLITS_PER_SEGMENT: usize = 4;
const LOW_INFORMATION_SEGMENT_MAX_MS: u64 = 2_500;
const LOW_INFORMATION_SEGMENT_MAX_WORDS: usize = 6;
const LOW_INFORMATION_NEIGHBOR_GAP_MS: u64 = 12_000;

#[cfg(feature = "speaker-embedding")]
use crate::transcribe::fbank::{FbankConfig, FbankExtractor};
#[cfg(feature = "speaker-embedding")]
use ndarray::Axis;
#[cfg(feature = "speaker-embedding-cuda")]
use ort::ep;
#[cfg(feature = "speaker-embedding")]
use ort::session::Session;
#[cfg(feature = "speaker-embedding")]
use ort::value::Tensor;

#[derive(Debug, Clone)]
struct SpeakerCentroid {
    id: u32,
    vector: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SpeakerWindow {
    start_ms: u64,
    end_ms: u64,
    speaker: SpeakerId,
    confidence: f32,
    embedding: Vec<f32>,
    initial_speaker: SpeakerId,
    initial_confidence: f32,
}

#[derive(Debug, Clone)]
struct UnknownCluster {
    start_idx: usize,
    end_idx: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SpeakerRun {
    start_ms: u64,
    end_ms: u64,
    speaker: SpeakerId,
    confidence: f32,
}

/// Local speaker embedding diarizer.
pub struct EmbeddingDiarizer {
    model_name: String,
    model_path: Option<PathBuf>,
    max_speakers: u32,
    min_segment_ms: u64,
    window_secs: f32,
    confident_threshold: f32,
    uncertain_threshold: f32,
    centroids: Vec<SpeakerCentroid>,
    #[cfg(feature = "speaker-embedding")]
    session: Option<std::sync::Mutex<Session>>,
    #[cfg(feature = "speaker-embedding")]
    fbank_extractor: FbankExtractor,
}

impl EmbeddingDiarizer {
    pub fn new(config: &DiarizationConfig) -> Self {
        Self {
            model_name: config.model.clone(),
            model_path: config.model_path.as_ref().map(PathBuf::from),
            max_speakers: config.max_speakers,
            min_segment_ms: config.min_segment_ms,
            window_secs: config.window_secs,
            confident_threshold: config.confident_threshold,
            uncertain_threshold: config.uncertain_threshold,
            centroids: Vec::new(),
            #[cfg(feature = "speaker-embedding")]
            session: None,
            #[cfg(feature = "speaker-embedding")]
            fbank_extractor: FbankExtractor::new(FbankConfig::default()),
        }
    }

    fn resolved_model_name(&self) -> &str {
        if self.model_name.trim().is_empty() {
            DEFAULT_MODEL_NAME
        } else {
            &self.model_name
        }
    }

    pub fn default_model_path(model_name: &str) -> PathBuf {
        match model_name {
            "wespeaker-resnet34" => Config::models_dir()
                .join("wespeaker-resnet34")
                .join("voxceleb_resnet34.onnx"),
            _ => Config::models_dir()
                .join("wespeaker-resnet221-lm")
                .join("voxceleb_resnet221_LM.onnx"),
        }
    }

    fn resolved_model_path(&self) -> PathBuf {
        self.model_path
            .clone()
            .unwrap_or_else(|| Self::default_model_path(self.resolved_model_name()))
    }

    pub fn model_exists(&self) -> bool {
        self.resolved_model_path().exists()
    }

    #[cfg(feature = "speaker-embedding")]
    pub fn load_model(&mut self) -> Result<(), String> {
        let path = self.resolved_model_path();
        let threads = num_cpus::get().min(4);
        let builder = Session::builder().map_err(|e| format!("ONNX session builder failed: {}", e))?;

        #[cfg(feature = "speaker-embedding-cuda")]
        let builder = builder
            .with_execution_providers([ep::CUDA::default().build()])
            .map_err(|e| format!("Failed to configure CUDA execution provider: {}", e))?;

        let session = builder
            .with_intra_threads(threads)
            .map_err(|e| format!("Failed to set threads: {}", e))?
            .commit_from_file(&path)
            .map_err(|e| {
                format!(
                    "Failed to load speaker embedding model from {:?}: {}",
                    path, e
                )
            })?;

        self.session = Some(std::sync::Mutex::new(session));
        tracing::info!(
            "Loaded speaker embedding model '{}' from {:?}{}",
            self.resolved_model_name(),
            path,
            if cfg!(feature = "speaker-embedding-cuda") {
                " with CUDA"
            } else {
                ""
            }
        );
        Ok(())
    }

    #[cfg(not(feature = "speaker-embedding"))]
    pub fn load_model(&mut self) -> Result<(), String> {
        Err("speaker-embedding feature not enabled".to_string())
    }

    #[cfg(feature = "speaker-embedding")]
    fn extract_embedding(&self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let window_samples = (self.window_secs * SAMPLE_RATE as f32) as usize;
        if samples.len() < self.min_segment_samples() {
            return Err("audio too short for embedding extraction".to_string());
        }

        let clip = if samples.len() > window_samples && window_samples > 0 {
            let start = (samples.len() - window_samples) / 2;
            &samples[start..start + window_samples]
        } else {
            samples
        };

        let mut features = self.fbank_extractor.extract(clip);
        if features.nrows() == 0 {
            return Err("audio too short for filterbank extraction".to_string());
        }

        // Match the chatbot project's per-bin cepstral mean normalization.
        if let Some(mean) = features.mean_axis(Axis(0)) {
            for mut row in features.rows_mut() {
                row -= &mean;
            }
        }

        let num_frames = features.nrows();
        let feat_dim = features.ncols();
        let (data, _offset) = features.into_raw_vec_and_offset();
        let input = Tensor::<f32>::from_array(([1usize, num_frames, feat_dim], data))
            .map_err(|e| format!("Failed to create embedding tensor: {}", e))?;

        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "speaker embedding model not loaded".to_string())?;
        let mut session = session
            .lock()
            .map_err(|e| format!("Failed to lock speaker embedding session: {}", e))?;

        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| format!("Speaker embedding inference failed: {}", e))?;

        let output = outputs
            .get("embs")
            .or_else(|| outputs.get("embedding"))
            .or_else(|| outputs.get("output"))
            .ok_or_else(|| "No embedding output found in ONNX model".to_string())?;

        let (_, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract embedding tensor: {}", e))?;

        Ok(data.to_vec())
    }

    fn min_segment_samples(&self) -> usize {
        ((self.min_segment_ms * SAMPLE_RATE) / 1000) as usize
    }

    fn speaker_hop_ms(&self) -> u64 {
        ((self.window_secs * 1000.0 / 4.0).round() as u64)
            .clamp(SPEAKER_WINDOW_HOP_MIN_MS, SPEAKER_WINDOW_HOP_MAX_MS)
    }

    fn unknown_cluster_threshold(&self) -> f32 {
        (self.confident_threshold + 0.15).clamp(0.45, 0.75)
    }

    fn best_matching_centroid(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        let mut best_idx = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            if centroid.vector.len() != embedding.len() {
                continue;
            }

            let sim = cosine_similarity(embedding, &centroid.vector);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        best_idx.map(|idx| (idx, best_sim.max(0.0)))
    }

    fn provisional_speaker_for_embedding(&self, embedding: &[f32]) -> (SpeakerId, f32) {
        if let Some((idx, sim)) = self.best_matching_centroid(embedding) {
            // Be conservative at the window-labeling stage. Lower-confidence reuse
            // happens later when we cluster unresolved runs, which gives the
            // second speaker a chance to form instead of being collapsed into
            // the first centroid immediately.
            if sim >= self.confident_threshold {
                return (SpeakerId::Auto(self.centroids[idx].id), sim);
            }
        }

        (SpeakerId::Remote, 0.0)
    }

    fn update_centroid(&mut self, idx: usize, embedding: &[f32]) {
        let Some(centroid) = self.centroids.get_mut(idx) else {
            return;
        };

        if centroid.vector.len() != embedding.len() {
            return;
        }

        centroid
            .vector
            .iter_mut()
            .zip(embedding.iter().copied())
            .for_each(|(old, new)| {
                *old = (1.0 - EMA_ALPHA) * *old + EMA_ALPHA * new;
            });
    }

    fn create_speaker(&mut self, embedding: &[f32]) -> Option<SpeakerId> {
        if (self.centroids.len() as u32) >= self.max_speakers {
            return None;
        }

        let id = self.centroids.len() as u32;
        self.centroids.push(SpeakerCentroid {
            id,
            vector: embedding.to_vec(),
        });
        Some(SpeakerId::Auto(id))
    }

    fn windows_overlap(start_ms: u64, end_ms: u64, other_start_ms: u64, other_end_ms: u64) -> u64 {
        end_ms
            .min(other_end_ms)
            .saturating_sub(start_ms.max(other_start_ms))
    }

    fn window_overlaps_transcript(
        start_ms: u64,
        end_ms: u64,
        transcript_segments: &[TranscriptSegment],
    ) -> bool {
        transcript_segments.iter().any(|segment| {
            Self::windows_overlap(start_ms, end_ms, segment.start_ms, segment.end_ms) > 0
        })
    }

    fn smooth_windows(&self, windows: &mut [SpeakerWindow]) {
        if windows.len() < 3 {
            return;
        }

        for _ in 0..2 {
            let snapshot = windows.to_vec();
            for idx in 1..snapshot.len() - 1 {
                let prev = &snapshot[idx - 1];
                let current = &snapshot[idx];
                let next = &snapshot[idx + 1];

                if prev.speaker == next.speaker
                    && current.speaker != prev.speaker
                    && current.confidence < self.confident_threshold
                {
                    windows[idx].speaker = prev.speaker.clone();
                    windows[idx].confidence = (prev.confidence + next.confidence) / 2.0;
                }
            }
        }
    }

    fn refresh_matched_centroids(&mut self, windows: &[SpeakerWindow]) {
        for window in windows {
            if window.initial_confidence < self.confident_threshold {
                continue;
            }

            let SpeakerId::Auto(id) = window.initial_speaker else {
                continue;
            };

            if let Some((idx, _)) = self
                .centroids
                .iter()
                .enumerate()
                .find(|(_, centroid)| centroid.id == id)
            {
                self.update_centroid(idx, &window.embedding);
            }
        }
    }

    fn speaker_scores_for_range(
        &self,
        start_ms: u64,
        end_ms: u64,
        windows: &[SpeakerWindow],
    ) -> HashMap<SpeakerId, f32> {
        if start_ms >= end_ms {
            return HashMap::new();
        }

        let mut scores: HashMap<SpeakerId, f32> = HashMap::new();

        for window in windows {
            let overlap = Self::windows_overlap(start_ms, end_ms, window.start_ms, window.end_ms);
            if overlap == 0 {
                continue;
            }

            *scores.entry(window.speaker.clone()).or_insert(0.0) +=
                overlap as f32 * window.confidence.max(0.1);
        }

        scores
    }

    fn dominant_speaker_for_range(
        &self,
        start_ms: u64,
        end_ms: u64,
        windows: &[SpeakerWindow],
    ) -> Option<(SpeakerId, f32)> {
        let scores = self.speaker_scores_for_range(start_ms, end_ms, windows);
        if scores.is_empty() {
            return None;
        }

        let total_score: f32 = scores.values().copied().sum();
        let (speaker, score) = scores
            .into_iter()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))?;

        let confidence = if total_score > 0.0 {
            (score / total_score).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Some((speaker, confidence))
    }

    fn segment_speaker_for_range(
        &self,
        start_ms: u64,
        end_ms: u64,
        windows: &[SpeakerWindow],
    ) -> (SpeakerId, f32) {
        let Some((speaker, confidence)) =
            self.dominant_speaker_for_range(start_ms, end_ms, windows)
        else {
            return (SpeakerId::Remote, 0.0);
        };

        let scores = self.speaker_scores_for_range(start_ms, end_ms, windows);
        let total_score: f32 = scores.values().copied().sum();
        let mut ranked: Vec<(SpeakerId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|(_, left), (_, right)| right.total_cmp(left));

        let competing_share = if total_score > 0.0 {
            ranked
                .iter()
                .skip(1)
                .map(|(_, score)| *score / total_score)
                .fold(0.0, f32::max)
        } else {
            0.0
        };

        if !matches!(speaker, SpeakerId::Remote)
            && competing_share >= AMBIGUOUS_COMPETITOR_SHARE
            && confidence < MIN_DOMINANT_SPEAKER_SHARE
        {
            return (SpeakerId::Remote, confidence);
        }

        (speaker, confidence)
    }

    fn speaker_runs_for_range(
        &self,
        start_ms: u64,
        end_ms: u64,
        windows: &[SpeakerWindow],
    ) -> Vec<SpeakerRun> {
        if start_ms >= end_ms {
            return vec![];
        }

        let mut boundaries = vec![start_ms, end_ms];
        for window in windows {
            let overlap = Self::windows_overlap(start_ms, end_ms, window.start_ms, window.end_ms);
            if overlap == 0 {
                continue;
            }
            boundaries.push(window.start_ms.max(start_ms));
            boundaries.push(window.end_ms.min(end_ms));
        }

        boundaries.sort_unstable();
        boundaries.dedup();

        let mut runs: Vec<SpeakerRun> = Vec::new();
        for interval in boundaries.windows(2) {
            let interval_start = interval[0];
            let interval_end = interval[1];
            if interval_end <= interval_start {
                continue;
            }

            let (speaker, confidence) =
                self.segment_speaker_for_range(interval_start, interval_end, windows);
            if let Some(last) = runs.last_mut() {
                if last.speaker == speaker {
                    let last_duration = last.duration_ms().max(1) as f32;
                    let interval_duration = (interval_end - interval_start).max(1) as f32;
                    last.end_ms = interval_end;
                    last.confidence = (last.confidence * last_duration
                        + confidence * interval_duration)
                        / (last_duration + interval_duration);
                    continue;
                }
            }

            runs.push(SpeakerRun {
                start_ms: interval_start,
                end_ms: interval_end,
                speaker,
                confidence,
            });
        }

        self.merge_transient_runs(&mut runs, start_ms, end_ms);
        runs
    }

    fn merge_transient_runs(
        &self,
        runs: &mut Vec<SpeakerRun>,
        range_start_ms: u64,
        range_end_ms: u64,
    ) {
        if runs.len() < 2 {
            return;
        }

        let total_duration = range_end_ms.saturating_sub(range_start_ms).max(1);

        loop {
            let mut changed = false;
            let mut idx = 0usize;

            while idx < runs.len() {
                let run_duration = runs[idx].duration_ms();
                let run_share = run_duration as f32 / total_duration as f32;
                let is_transient = run_duration < TRANSIENT_SPEAKER_RUN_MS
                    || (run_share < 0.08 && runs[idx].confidence < self.confident_threshold);

                if !is_transient || runs.len() == 1 {
                    idx += 1;
                    continue;
                }

                if idx > 0 && idx + 1 < runs.len() && runs[idx - 1].speaker == runs[idx + 1].speaker
                {
                    let merged_confidence =
                        (runs[idx - 1].confidence + runs[idx + 1].confidence) / 2.0;
                    runs[idx - 1].end_ms = runs[idx + 1].end_ms;
                    runs[idx - 1].confidence = merged_confidence;
                    runs.remove(idx + 1);
                    runs.remove(idx);
                    changed = true;
                    idx = idx.saturating_sub(1);
                    continue;
                }

                let merge_left = if idx == 0 {
                    false
                } else if idx + 1 >= runs.len() {
                    true
                } else {
                    runs[idx - 1].duration_ms() >= runs[idx + 1].duration_ms()
                };

                if merge_left {
                    runs[idx - 1].end_ms = runs[idx].end_ms;
                    runs[idx - 1].confidence =
                        (runs[idx - 1].confidence + runs[idx].confidence) / 2.0;
                    runs.remove(idx);
                    changed = true;
                    idx = idx.saturating_sub(1);
                } else {
                    runs[idx + 1].start_ms = runs[idx].start_ms;
                    runs[idx + 1].confidence =
                        (runs[idx + 1].confidence + runs[idx].confidence) / 2.0;
                    runs.remove(idx);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
    }

    fn sentence_chunks(text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    chunks.push(trimmed.to_string());
                }
                current.clear();
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }

        if chunks.is_empty() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
        }

        chunks
    }

    fn split_text_by_runs(&self, text: &str, runs: &[SpeakerRun]) -> Option<Vec<String>> {
        if runs.len() <= 1 {
            return Some(vec![text.trim().to_string()]);
        }

        let sentence_chunks = Self::sentence_chunks(text);
        if sentence_chunks.len() < runs.len() {
            return None;
        }

        let total_duration: u64 = runs.iter().map(SpeakerRun::duration_ms).sum();
        if total_duration == 0 {
            return Some(vec![text.trim().to_string()]);
        }

        let mut parts = Vec::with_capacity(runs.len());
        let mut sentence_start = 0usize;
        let mut consumed_duration = 0u64;

        for (idx, run) in runs.iter().enumerate() {
            if idx + 1 == runs.len() {
                parts.push(sentence_chunks[sentence_start..].join(" "));
                break;
            }

            consumed_duration += run.duration_ms();
            let remaining_runs = runs.len() - idx - 1;
            let min_end = sentence_start + 1;
            let max_end = sentence_chunks.len().saturating_sub(remaining_runs);
            let target = ((sentence_chunks.len() as f32 * consumed_duration as f32
                / total_duration as f32)
                .round() as usize)
                .clamp(min_end, max_end.max(min_end));
            parts.push(sentence_chunks[sentence_start..target].join(" "));
            sentence_start = target;
        }

        Some(parts)
    }

    fn split_segment_by_speaker_runs(
        &self,
        segment: &TranscriptSegment,
        windows: &[SpeakerWindow],
    ) -> Vec<DiarizedSegment> {
        let mut runs = self.speaker_runs_for_range(segment.start_ms, segment.end_ms, windows);
        if runs.len() <= 1 {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        }

        let segment_duration = segment.duration_ms().max(1);
        runs.retain(|run| {
            let share = run.duration_ms() as f32 / segment_duration as f32;
            run.duration_ms() >= MIN_SPLIT_RUN_MS || share >= MIN_SPLIT_RUN_SHARE
        });

        if runs.len() <= 1 {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        }

        if runs.len() > MAX_SPEAKER_SPLITS_PER_SEGMENT {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        }

        let Some(parts) = self.split_text_by_runs(&segment.text, &runs) else {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        };

        if parts.len() != runs.len() {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        }

        let mut diarized_segments = Vec::with_capacity(runs.len());
        for (run, text) in runs.into_iter().zip(parts.into_iter()) {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            diarized_segments.push(DiarizedSegment {
                speaker: run.speaker,
                start_ms: run.start_ms.max(segment.start_ms),
                end_ms: run.end_ms.min(segment.end_ms),
                text: text.to_string(),
                confidence: run.confidence,
            });
        }

        if diarized_segments.is_empty() {
            let (speaker, confidence) =
                self.segment_speaker_for_range(segment.start_ms, segment.end_ms, windows);
            return vec![DiarizedSegment {
                speaker,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence,
            }];
        }

        diarized_segments
    }

    fn segment_word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn is_low_information_segment(segment: &DiarizedSegment) -> bool {
        let duration_ms = segment.end_ms.saturating_sub(segment.start_ms);
        let word_count = Self::segment_word_count(&segment.text);
        duration_ms <= LOW_INFORMATION_SEGMENT_MAX_MS
            && word_count <= LOW_INFORMATION_SEGMENT_MAX_WORDS
    }

    fn absorb_low_information_segments(&self, segments: &mut [DiarizedSegment]) {
        if segments.len() < 3 {
            return;
        }

        for idx in 1..segments.len() - 1 {
            if !Self::is_low_information_segment(&segments[idx]) {
                continue;
            }

            let prev_speaker = segments[idx - 1].speaker.clone();
            let prev_end_ms = segments[idx - 1].end_ms;
            let prev_confidence = segments[idx - 1].confidence;
            let next_speaker = segments[idx + 1].speaker.clone();
            let next_start_ms = segments[idx + 1].start_ms;
            let next_confidence = segments[idx + 1].confidence;

            if prev_speaker != next_speaker {
                continue;
            }

            if !matches!(prev_speaker, SpeakerId::Auto(_)) {
                continue;
            }

            let prev_gap = segments[idx].start_ms.saturating_sub(prev_end_ms);
            let next_gap = next_start_ms.saturating_sub(segments[idx].end_ms);
            if prev_gap > LOW_INFORMATION_NEIGHBOR_GAP_MS
                || next_gap > LOW_INFORMATION_NEIGHBOR_GAP_MS
            {
                continue;
            }

            if prev_confidence < self.confident_threshold
                || next_confidence < self.confident_threshold
            {
                continue;
            }

            segments[idx].speaker = prev_speaker;
            segments[idx].confidence = segments[idx]
                .confidence
                .max(prev_confidence.min(next_confidence));
        }
    }

    fn cluster_unknown_windows(&self, windows: &[SpeakerWindow]) -> Vec<UnknownCluster> {
        if windows.is_empty() {
            return vec![];
        }

        let similarity_threshold = self.unknown_cluster_threshold();
        let mut clusters: Vec<UnknownCluster> = Vec::new();

        for (offset, window) in windows.iter().enumerate() {
            if let Some(last) = clusters.last_mut() {
                let similarity = cosine_similarity(&window.embedding, &last.embedding);
                if similarity >= similarity_threshold {
                    let current_count = last.window_count() as f32;
                    for (avg, new) in last.embedding.iter_mut().zip(window.embedding.iter()) {
                        *avg = (*avg * current_count + *new) / (current_count + 1.0);
                    }
                    last.end_idx = offset + 1;
                    continue;
                }
            }

            clusters.push(UnknownCluster {
                start_idx: offset,
                end_idx: offset + 1,
                embedding: window.embedding.clone(),
            });
        }

        clusters
    }

    fn promote_unknown_runs(&mut self, windows: &mut [SpeakerWindow]) {
        let mut run_start = 0usize;

        while run_start < windows.len() {
            if windows[run_start].speaker != SpeakerId::Remote {
                run_start += 1;
                continue;
            }

            let mut run_end = run_start + 1;
            while run_end < windows.len() && windows[run_end].speaker == SpeakerId::Remote {
                run_end += 1;
            }

            let run_clusters = self.cluster_unknown_windows(&windows[run_start..run_end]);
            for cluster in run_clusters {
                let abs_start = run_start + cluster.start_idx;
                let abs_end = run_start + cluster.end_idx;
                let cluster_duration_ms = windows[abs_end - 1]
                    .end_ms
                    .saturating_sub(windows[abs_start].start_ms);

                let assignment =
                    if let Some((idx, sim)) = self.best_matching_centroid(&cluster.embedding) {
                        if sim >= self.uncertain_threshold {
                            let speaker_id = self.centroids[idx].id;
                            if sim >= self.confident_threshold {
                                self.update_centroid(idx, &cluster.embedding);
                            }
                            Some((SpeakerId::Auto(speaker_id), sim))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                let (speaker, confidence) = assignment.unwrap_or_else(|| {
                    if cluster.window_count() >= NEW_SPEAKER_MIN_WINDOWS
                        && cluster_duration_ms >= NEW_SPEAKER_MIN_RUN_MS
                    {
                        self.create_speaker(&cluster.embedding)
                            .map(|speaker| (speaker, NEW_SPEAKER_CONFIDENCE))
                            .unwrap_or((SpeakerId::Remote, 0.0))
                    } else {
                        (SpeakerId::Remote, 0.0)
                    }
                });

                for window in &mut windows[abs_start..abs_end] {
                    window.speaker = speaker.clone();
                    window.confidence = confidence;
                }
            }

            run_start = run_end;
        }
    }

    #[cfg(feature = "speaker-embedding")]
    fn build_speaker_windows(
        &mut self,
        samples: &[f32],
        chunk_start_ms: u64,
        transcript_segments: &[TranscriptSegment],
    ) -> Vec<SpeakerWindow> {
        if samples.is_empty()
            || transcript_segments.is_empty()
            || samples.len() < self.min_segment_samples()
        {
            return vec![];
        }

        let window_samples =
            ((self.window_secs * SAMPLE_RATE as f32) as usize).max(self.min_segment_samples());
        let hop_samples = ((self.speaker_hop_ms() * SAMPLE_RATE) / 1000) as usize;
        let mut starts = Vec::new();

        if samples.len() <= window_samples {
            starts.push(0usize);
        } else {
            let max_start = samples.len() - window_samples;
            let mut start = 0usize;
            while start <= max_start {
                starts.push(start);
                start += hop_samples.max(1);
            }
            if starts.last().copied() != Some(max_start) {
                starts.push(max_start);
            }
        }

        let mut windows = Vec::new();
        for start_sample in starts {
            let end_sample = (start_sample + window_samples).min(samples.len());
            let start_ms = chunk_start_ms + (start_sample as u64 * 1000) / SAMPLE_RATE;
            let end_ms = chunk_start_ms + (end_sample as u64 * 1000) / SAMPLE_RATE;
            if !Self::window_overlaps_transcript(start_ms, end_ms, transcript_segments) {
                continue;
            }

            match self.extract_embedding(&samples[start_sample..end_sample]) {
                Ok(embedding) => {
                    let (speaker, confidence) = self.provisional_speaker_for_embedding(&embedding);
                    let initial_speaker = speaker.clone();
                    windows.push(SpeakerWindow {
                        start_ms,
                        end_ms,
                        speaker,
                        confidence,
                        embedding,
                        initial_speaker,
                        initial_confidence: confidence,
                    });
                }
                Err(e) => {
                    tracing::debug!(
                        "Speaker embedding extraction failed for loopback analysis window: {}",
                        e
                    );
                }
            }
        }

        windows
    }

    fn diarize_loopback_segments(
        &mut self,
        samples: &[f32],
        chunk_start_ms: u64,
        transcript_segments: &[TranscriptSegment],
    ) -> Vec<DiarizedSegment> {
        let mut speaker_windows =
            self.build_speaker_windows(samples, chunk_start_ms, transcript_segments);
        self.smooth_windows(&mut speaker_windows);
        self.refresh_matched_centroids(&speaker_windows);
        self.promote_unknown_runs(&mut speaker_windows);
        self.smooth_windows(&mut speaker_windows);

        let mut diarized_segments = Vec::new();
        for segment in transcript_segments {
            diarized_segments.extend(self.split_segment_by_speaker_runs(segment, &speaker_windows));
        }
        self.absorb_low_information_segments(&mut diarized_segments);
        diarized_segments
    }
}

impl Diarizer for EmbeddingDiarizer {
    fn diarize(
        &mut self,
        samples: &[f32],
        chunk_start_ms: u64,
        source: AudioSource,
        transcript_segments: &[TranscriptSegment],
    ) -> Vec<DiarizedSegment> {
        match source {
            AudioSource::Microphone => transcript_segments
                .iter()
                .map(|seg| DiarizedSegment {
                    speaker: SpeakerId::You,
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    text: seg.text.clone(),
                    confidence: 1.0,
                })
                .collect(),
            AudioSource::Unknown => transcript_segments
                .iter()
                .map(|seg| DiarizedSegment {
                    speaker: SpeakerId::Unknown,
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    text: seg.text.clone(),
                    confidence: 0.0,
                })
                .collect(),
            AudioSource::Loopback => {
                self.diarize_loopback_segments(samples, chunk_start_ms, transcript_segments)
            }
        }
    }

    fn name(&self) -> &'static str {
        "embedding"
    }

    fn reset(&mut self) {
        self.centroids.clear();
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

impl UnknownCluster {
    fn window_count(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx)
    }
}

impl SpeakerRun {
    fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_path_uses_large_wespeaker_model() {
        let path = EmbeddingDiarizer::default_model_path(DEFAULT_MODEL_NAME);
        assert!(path.ends_with("voxceleb_resnet221_LM.onnx"));
    }

    #[test]
    fn test_best_matching_centroid_reuses_existing_speaker() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);

        let speaker = diarizer.create_speaker(&[1.0, 0.0, 0.0]).expect("speaker");
        let (idx, score) = diarizer
            .best_matching_centroid(&[0.95, 0.05, 0.0])
            .expect("match");

        assert_eq!(speaker, SpeakerId::Auto(0));
        assert_eq!(diarizer.centroids[idx].id, 0);
        assert!(score >= diarizer.uncertain_threshold);
    }

    #[test]
    fn test_provisional_speaker_requires_confident_match() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        diarizer.create_speaker(&[1.0, 0.0, 0.0]).expect("speaker");

        let (speaker, confidence) = diarizer.provisional_speaker_for_embedding(&[0.2, 0.98, 0.0]);

        assert_eq!(speaker, SpeakerId::Remote);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_refresh_matched_centroids_updates_existing_speaker() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        diarizer.create_speaker(&[1.0, 0.0, 0.0]).expect("speaker");

        let windows = vec![SpeakerWindow {
            start_ms: 0,
            end_ms: 1_500,
            speaker: SpeakerId::Auto(0),
            confidence: 0.9,
            embedding: vec![0.8, 0.2, 0.0],
            initial_speaker: SpeakerId::Auto(0),
            initial_confidence: 0.9,
        }];

        diarizer.refresh_matched_centroids(&windows);

        assert!(diarizer.centroids[0].vector[1] > 0.0);
    }

    #[test]
    fn test_dominant_speaker_for_range_prefers_weighted_overlap() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let windows = vec![
            SpeakerWindow {
                start_ms: 0,
                end_ms: 1_500,
                speaker: SpeakerId::Auto(0),
                confidence: 0.9,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Auto(0),
                initial_confidence: 0.9,
            },
            SpeakerWindow {
                start_ms: 1_000,
                end_ms: 2_500,
                speaker: SpeakerId::Auto(1),
                confidence: 0.4,
                embedding: vec![0.0, 1.0],
                initial_speaker: SpeakerId::Auto(1),
                initial_confidence: 0.4,
            },
        ];

        let (speaker, confidence) = diarizer
            .dominant_speaker_for_range(500, 2_000, &windows)
            .expect("speaker");

        assert_eq!(speaker, SpeakerId::Auto(0));
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_promote_unknown_runs_requires_multiple_windows() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        let mut windows = vec![SpeakerWindow {
            start_ms: 0,
            end_ms: 1_500,
            speaker: SpeakerId::Remote,
            confidence: 0.0,
            embedding: vec![1.0, 0.0],
            initial_speaker: SpeakerId::Remote,
            initial_confidence: 0.0,
        }];

        diarizer.promote_unknown_runs(&mut windows);

        assert_eq!(windows[0].speaker, SpeakerId::Remote);
        assert!(diarizer.centroids.is_empty());
    }

    #[test]
    fn test_promote_unknown_runs_reuses_returning_speaker() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        let mut windows = vec![
            SpeakerWindow {
                start_ms: 0,
                end_ms: 1_500,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
            SpeakerWindow {
                start_ms: 375,
                end_ms: 1_875,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![0.98, 0.02],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
            SpeakerWindow {
                start_ms: 750,
                end_ms: 2_250,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![0.0, 1.0],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
            SpeakerWindow {
                start_ms: 1_125,
                end_ms: 2_625,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![0.02, 0.98],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
            SpeakerWindow {
                start_ms: 1_500,
                end_ms: 3_000,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
            SpeakerWindow {
                start_ms: 1_875,
                end_ms: 3_375,
                speaker: SpeakerId::Remote,
                confidence: 0.0,
                embedding: vec![0.97, 0.03],
                initial_speaker: SpeakerId::Remote,
                initial_confidence: 0.0,
            },
        ];

        diarizer.promote_unknown_runs(&mut windows);

        assert_eq!(diarizer.centroids.len(), 2);
        assert_eq!(windows[0].speaker, SpeakerId::Auto(0));
        assert_eq!(windows[1].speaker, SpeakerId::Auto(0));
        assert_eq!(windows[2].speaker, SpeakerId::Auto(1));
        assert_eq!(windows[3].speaker, SpeakerId::Auto(1));
        assert_eq!(windows[4].speaker, SpeakerId::Auto(0));
        assert_eq!(windows[5].speaker, SpeakerId::Auto(0));
    }

    #[test]
    fn test_segment_speaker_for_range_marks_ambiguous_overlap_as_remote() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let windows = vec![
            SpeakerWindow {
                start_ms: 0,
                end_ms: 1_500,
                speaker: SpeakerId::Auto(0),
                confidence: 0.9,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Auto(0),
                initial_confidence: 0.9,
            },
            SpeakerWindow {
                start_ms: 1_000,
                end_ms: 2_500,
                speaker: SpeakerId::Auto(1),
                confidence: 0.85,
                embedding: vec![0.0, 1.0],
                initial_speaker: SpeakerId::Auto(1),
                initial_confidence: 0.85,
            },
        ];

        let (speaker, confidence) = diarizer.segment_speaker_for_range(500, 2_000, &windows);

        assert_eq!(speaker, SpeakerId::Remote);
        assert!(confidence > 0.4);
    }

    #[test]
    fn test_reset_clears_centroids() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        diarizer.create_speaker(&[1.0, 0.0, 0.0]).expect("speaker");

        diarizer.reset();

        assert!(diarizer.centroids.is_empty());
    }

    #[test]
    fn test_refresh_matched_centroids_ignores_smoothed_low_confidence_windows() {
        let config = DiarizationConfig::default();
        let mut diarizer = EmbeddingDiarizer::new(&config);
        diarizer.create_speaker(&[1.0, 0.0, 0.0]).expect("speaker");

        let baseline = diarizer.centroids[0].vector.clone();
        let windows = vec![SpeakerWindow {
            start_ms: 0,
            end_ms: 1_500,
            speaker: SpeakerId::Auto(0),
            confidence: 0.95,
            embedding: vec![0.0, 1.0, 0.0],
            initial_speaker: SpeakerId::Remote,
            initial_confidence: 0.0,
        }];

        diarizer.refresh_matched_centroids(&windows);

        assert_eq!(diarizer.centroids[0].vector, baseline);
    }

    #[test]
    fn test_split_segment_by_speaker_runs_splits_clear_two_speaker_turn() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let segment = TranscriptSegment {
            id: 0,
            start_ms: 0,
            end_ms: 4_000,
            text: "First speaker says hello. Second speaker answers back.".to_string(),
            source: AudioSource::Loopback,
            speaker_id: None,
            speaker_label: None,
            confidence: None,
            chunk_id: 0,
        };
        let windows = vec![
            SpeakerWindow {
                start_ms: 0,
                end_ms: 2_000,
                speaker: SpeakerId::Auto(0),
                confidence: 0.95,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Auto(0),
                initial_confidence: 0.95,
            },
            SpeakerWindow {
                start_ms: 2_000,
                end_ms: 4_000,
                speaker: SpeakerId::Auto(1),
                confidence: 0.96,
                embedding: vec![0.0, 1.0],
                initial_speaker: SpeakerId::Auto(1),
                initial_confidence: 0.96,
            },
        ];

        let diarized = diarizer.split_segment_by_speaker_runs(&segment, &windows);

        assert_eq!(diarized.len(), 2);
        assert_eq!(diarized[0].speaker, SpeakerId::Auto(0));
        assert_eq!(diarized[1].speaker, SpeakerId::Auto(1));
        assert!(diarized[0].text.contains("First speaker says hello"));
        assert!(diarized[1].text.contains("Second speaker answers back"));
    }

    #[test]
    fn test_split_segment_by_speaker_runs_does_not_split_without_sentence_evidence() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let segment = TranscriptSegment {
            id: 0,
            start_ms: 0,
            end_ms: 4_000,
            text: "First speaker says hello and second speaker answers back".to_string(),
            source: AudioSource::Loopback,
            speaker_id: None,
            speaker_label: None,
            confidence: None,
            chunk_id: 0,
        };
        let windows = vec![
            SpeakerWindow {
                start_ms: 0,
                end_ms: 2_000,
                speaker: SpeakerId::Auto(0),
                confidence: 0.95,
                embedding: vec![1.0, 0.0],
                initial_speaker: SpeakerId::Auto(0),
                initial_confidence: 0.95,
            },
            SpeakerWindow {
                start_ms: 2_000,
                end_ms: 4_000,
                speaker: SpeakerId::Auto(1),
                confidence: 0.96,
                embedding: vec![0.0, 1.0],
                initial_speaker: SpeakerId::Auto(1),
                initial_confidence: 0.96,
            },
        ];

        let diarized = diarizer.split_segment_by_speaker_runs(&segment, &windows);

        assert_eq!(diarized.len(), 1);
        assert_eq!(diarized[0].speaker, SpeakerId::Remote);
        assert_eq!(diarized[0].text, segment.text);
    }

    #[test]
    fn test_absorb_low_information_segments_relabels_sandwiched_backchannel() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let mut segments = vec![
            DiarizedSegment {
                speaker: SpeakerId::Auto(0),
                start_ms: 0,
                end_ms: 5_000,
                text: "Longer response from the main speaker.".to_string(),
                confidence: 0.95,
            },
            DiarizedSegment {
                speaker: SpeakerId::Auto(1),
                start_ms: 5_100,
                end_ms: 5_900,
                text: "Yeah.".to_string(),
                confidence: 1.0,
            },
            DiarizedSegment {
                speaker: SpeakerId::Auto(0),
                start_ms: 6_000,
                end_ms: 11_000,
                text: "Another longer response from the main speaker.".to_string(),
                confidence: 0.94,
            },
        ];

        diarizer.absorb_low_information_segments(&mut segments);

        assert_eq!(segments[1].speaker, SpeakerId::Auto(0));
    }

    #[test]
    fn test_absorb_low_information_segments_preserves_distinct_longer_turn() {
        let diarizer = EmbeddingDiarizer::new(&DiarizationConfig::default());
        let mut segments = vec![
            DiarizedSegment {
                speaker: SpeakerId::Auto(0),
                start_ms: 0,
                end_ms: 5_000,
                text: "Longer response from the main speaker.".to_string(),
                confidence: 0.95,
            },
            DiarizedSegment {
                speaker: SpeakerId::Auto(1),
                start_ms: 5_100,
                end_ms: 10_500,
                text: "A materially different turn with enough words to stand on its own."
                    .to_string(),
                confidence: 0.97,
            },
            DiarizedSegment {
                speaker: SpeakerId::Auto(0),
                start_ms: 10_700,
                end_ms: 15_000,
                text: "Another longer response from the main speaker.".to_string(),
                confidence: 0.94,
            },
        ];

        diarizer.absorb_low_information_segments(&mut segments);

        assert_eq!(segments[1].speaker, SpeakerId::Auto(1));
    }
}
