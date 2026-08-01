//! Audio chunk processor for meeting transcription
//!
//! Handles splitting continuous audio into chunks, applying VAD,
//! and coordinating transcription.

use crate::error::TranscribeError;
use crate::meeting::data::{AudioSource, TranscriptSegment};
use crate::transcribe::Transcriber;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIN_CHUNK_SPEECH_MS: u32 = 240;
const TRANSCRIPT_END_SILENCE_MS: u32 = 750;
const TRANSCRIPT_PADDING_MS: u32 = 180;
const TRANSCRIPT_MAX_SEGMENT_MS: u32 = 12_000;

/// Configuration for chunk processing
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Duration of each audio chunk in seconds
    pub chunk_duration_secs: u32,
    /// Minimum audio level to consider as speech (0.0 - 1.0)
    pub vad_threshold: f32,
    /// Sample rate (expected 16000 Hz)
    pub sample_rate: u32,
    /// Minimum chunk duration to process (in seconds)
    pub min_chunk_duration_secs: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_duration_secs: 30,
            vad_threshold: 0.01,
            sample_rate: 16000,
            min_chunk_duration_secs: 0.5,
        }
    }
}

/// Audio buffer for a chunk being recorded
#[derive(Debug)]
pub struct ChunkBuffer {
    /// Audio samples (mono, f32, 16kHz)
    samples: Vec<f32>,
    /// Start time of this chunk
    started_at: Instant,
    /// Chunk ID
    chunk_id: u32,
    /// Audio source
    source: AudioSource,
    /// Start time offset in milliseconds (relative to meeting start)
    start_offset_ms: u64,
}

impl ChunkBuffer {
    /// Create a new chunk buffer
    pub fn new(chunk_id: u32, source: AudioSource, start_offset_ms: u64) -> Self {
        Self {
            samples: Vec::with_capacity(16000 * 30), // Pre-allocate for 30 seconds
            started_at: Instant::now(),
            chunk_id,
            source,
            start_offset_ms,
        }
    }

    /// Add audio samples to the buffer
    pub fn add_samples(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    /// Get the duration of audio in seconds
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / 16000.0
    }

    /// Get the elapsed wall-clock time
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Take ownership of the samples, leaving buffer empty
    pub fn take_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }

    /// Check if buffer has any audio
    pub fn has_audio(&self) -> bool {
        !self.samples.is_empty()
    }
}

/// Voice Activity Detection (VAD)
///
/// Simple energy-based VAD for filtering silent chunks.
/// Phase 3+ will use more sophisticated ML-based VAD.
pub struct VoiceActivityDetector {
    threshold: f32,
    sample_rate: u32,
    /// Window size for energy calculation in milliseconds
    window_ms: u32,
}

impl VoiceActivityDetector {
    /// Create a new VAD with the given threshold
    pub fn new(threshold: f32, sample_rate: u32) -> Self {
        Self {
            threshold,
            sample_rate,
            window_ms: 30,
        }
    }

    /// Check if the audio contains speech
    pub fn contains_speech(&self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }

        // Calculate RMS energy over windows
        let window_size = (self.sample_rate * self.window_ms / 1000) as usize;
        if window_size == 0 {
            return false;
        }

        let mut speech_frames = 0;
        for chunk in samples.chunks(window_size) {
            let rms = Self::calculate_rms(chunk);
            if rms > self.threshold {
                speech_frames += 1;
            }
        }

        let min_speech_frames = MIN_CHUNK_SPEECH_MS.div_ceil(self.window_ms) as usize;
        speech_frames >= min_speech_frames
    }

    /// Calculate RMS energy of samples
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Get speech segments with their boundaries
    ///
    /// Returns a list of (start_sample, end_sample) tuples for speech regions.
    pub fn detect_speech_segments(&self, samples: &[f32]) -> Vec<(usize, usize)> {
        let window_size = (self.sample_rate * self.window_ms / 1000) as usize;
        if window_size == 0 || samples.is_empty() {
            return vec![];
        }

        let mut raw_segments = vec![];
        let mut in_speech = false;
        let mut speech_start = 0usize;
        let mut silence_count = 0;
        let hangover = TRANSCRIPT_END_SILENCE_MS.div_ceil(self.window_ms) as usize;
        let padding = TRANSCRIPT_PADDING_MS.div_ceil(self.window_ms) as usize * window_size;
        let max_segment_samples =
            TRANSCRIPT_MAX_SEGMENT_MS as usize * self.sample_rate as usize / 1000;
        let mut last_speech_end = 0usize;

        for (i, chunk) in samples.chunks(window_size).enumerate() {
            let rms = Self::calculate_rms(chunk);
            let is_speech = rms > self.threshold;

            if is_speech {
                silence_count = 0;
                if !in_speech {
                    in_speech = true;
                    speech_start = i * window_size;
                }
                last_speech_end = ((i + 1) * window_size).min(samples.len());
            } else if in_speech {
                silence_count += 1;
                if silence_count >= hangover {
                    // End of speech segment
                    raw_segments.push((speech_start, last_speech_end));
                    in_speech = false;
                    silence_count = 0;
                }
            }
        }

        // Handle speech that extends to the end
        if in_speech {
            raw_segments.push((speech_start, samples.len().max(last_speech_end)));
        }

        let mut segments: Vec<(usize, usize)> = Vec::new();
        for (start, end) in raw_segments {
            let padded_start = start.saturating_sub(padding);
            let padded_end = (end + padding).min(samples.len());

            if let Some((last_start, last_end)) = segments.last_mut() {
                if padded_start <= *last_end {
                    *last_start = (*last_start).min(padded_start);
                    *last_end = (*last_end).max(padded_end);
                    continue;
                }
            }

            segments.push((padded_start, padded_end));
        }

        split_long_segments(segments, max_segment_samples)
    }
}

fn split_long_segments(
    segments: Vec<(usize, usize)>,
    max_segment_samples: usize,
) -> Vec<(usize, usize)> {
    if max_segment_samples == 0 {
        return segments;
    }

    let mut result = Vec::new();
    for (start, end) in segments {
        let mut current_start = start;
        while end.saturating_sub(current_start) > max_segment_samples {
            let split_end = current_start + max_segment_samples;
            result.push((current_start, split_end));
            current_start = split_end;
        }
        if end > current_start {
            result.push((current_start, end));
        }
    }

    result
}

/// Processed chunk result
#[derive(Debug)]
pub struct ProcessedChunk {
    /// Chunk ID
    pub chunk_id: u32,
    /// Absolute start time of the chunk in milliseconds from meeting start
    pub chunk_start_ms: u64,
    /// Transcript segments from this chunk
    pub segments: Vec<TranscriptSegment>,
    /// Original audio samples for this chunk
    pub samples: Vec<f32>,
    /// Original audio duration in milliseconds
    pub audio_duration_ms: u64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Chunk processor
///
/// Coordinates audio buffering, VAD, and transcription for meeting mode.
pub struct ChunkProcessor {
    config: ChunkConfig,
    vad: VoiceActivityDetector,
    transcriber: Arc<dyn Transcriber>,
    next_segment_id: u32,
}

impl ChunkProcessor {
    /// Create a new chunk processor
    pub fn new(config: ChunkConfig, transcriber: Arc<dyn Transcriber>) -> Self {
        Self::new_with_segment_id(config, transcriber, 0)
    }

    /// Create a new chunk processor starting from a specific segment ID
    pub fn new_with_segment_id(
        config: ChunkConfig,
        transcriber: Arc<dyn Transcriber>,
        next_segment_id: u32,
    ) -> Self {
        let vad = VoiceActivityDetector::new(config.vad_threshold, config.sample_rate);
        Self {
            config,
            vad,
            transcriber,
            next_segment_id,
        }
    }

    /// Get the next segment ID after processing.
    pub fn next_segment_id(&self) -> u32 {
        self.next_segment_id
    }

    /// Process a completed chunk of audio
    ///
    /// Applies VAD, transcribes speech regions, and returns transcript segments.
    pub fn process_chunk(
        &mut self,
        buffer: ChunkBuffer,
    ) -> Result<ProcessedChunk, TranscribeError> {
        let start_time = Instant::now();
        let chunk_id = buffer.chunk_id;
        let source = buffer.source;
        let start_offset_ms = buffer.start_offset_ms;

        let samples = buffer.samples;
        let audio_duration_ms = (samples.len() as f64 / 16000.0 * 1000.0) as u64;

        // Skip if too short
        let min_samples = (self.config.min_chunk_duration_secs * 16000.0) as usize;
        if samples.len() < min_samples {
            tracing::debug!(
                "Chunk {} too short ({:.2}s), skipping",
                chunk_id,
                samples.len() as f32 / 16000.0
            );
            return Ok(ProcessedChunk {
                chunk_id,
                chunk_start_ms: start_offset_ms,
                segments: vec![],
                samples,
                audio_duration_ms,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Check for speech
        if !self.vad.contains_speech(&samples) {
            tracing::debug!("Chunk {} has no speech, skipping", chunk_id);
            return Ok(ProcessedChunk {
                chunk_id,
                chunk_start_ms: start_offset_ms,
                segments: vec![],
                samples,
                audio_duration_ms,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        let speech_segments = self.vad.detect_speech_segments(&samples);
        let mut segments = vec![];
        let min_samples =
            (self.config.min_chunk_duration_secs * self.config.sample_rate as f32) as usize;

        for (seg_start, seg_end) in speech_segments {
            if seg_end <= seg_start {
                continue;
            }

            let segment_samples = &samples[seg_start..seg_end];
            if segment_samples.len() < min_samples {
                continue;
            }

            tracing::info!(
                "Transcribing chunk {} speech region ({:.1}s of audio)",
                chunk_id,
                segment_samples.len() as f32 / self.config.sample_rate as f32
            );

            let text = self.transcriber.transcribe(segment_samples)?;
            if text.trim().is_empty() {
                continue;
            }

            let segment_id = self.next_segment_id;
            self.next_segment_id += 1;

            let seg_start_ms =
                start_offset_ms + (seg_start as u64 * 1000) / self.config.sample_rate as u64;
            let seg_end_ms =
                start_offset_ms + (seg_end as u64 * 1000) / self.config.sample_rate as u64;

            let mut segment = TranscriptSegment::new(
                segment_id,
                seg_start_ms,
                seg_end_ms,
                text.trim().to_string(),
                chunk_id,
            );
            segment.source = source;
            segments.push(segment);
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        tracing::debug!("Chunk {} processed in {}ms", chunk_id, processing_time_ms);

        Ok(ProcessedChunk {
            chunk_id,
            chunk_start_ms: start_offset_ms,
            segments,
            samples,
            audio_duration_ms,
            processing_time_ms,
        })
    }

    /// Check if a chunk buffer is ready for processing
    pub fn is_chunk_ready(&self, buffer: &ChunkBuffer) -> bool {
        buffer.duration_secs() >= self.config.chunk_duration_secs as f32
    }

    /// Create a new chunk buffer
    pub fn new_buffer(
        &self,
        chunk_id: u32,
        source: AudioSource,
        start_offset_ms: u64,
    ) -> ChunkBuffer {
        ChunkBuffer::new(chunk_id, source, start_offset_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TranscribeError;
    use std::sync::{Arc, Mutex};

    fn create_test_samples(duration_secs: f32, frequency_hz: f32, amplitude: f32) -> Vec<f32> {
        let sample_rate = 16000.0;
        let num_samples = (duration_secs * sample_rate) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                amplitude * (2.0 * std::f32::consts::PI * frequency_hz * t).sin()
            })
            .collect()
    }

    fn create_silent_samples(duration_secs: f32) -> Vec<f32> {
        let num_samples = (duration_secs * 16000.0) as usize;
        vec![0.0; num_samples]
    }

    #[derive(Default)]
    struct MockTranscriber {
        calls: Mutex<Vec<usize>>,
    }

    impl MockTranscriber {
        fn observed_calls(&self) -> Vec<usize> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Transcriber for MockTranscriber {
        fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
            let mut calls = self.calls.lock().unwrap();
            calls.push(samples.len());
            Ok(format!("segment {}", calls.len()))
        }
    }

    #[test]
    fn test_chunk_buffer_duration() {
        let mut buffer = ChunkBuffer::new(0, AudioSource::Microphone, 0);
        buffer.add_samples(&vec![0.0; 16000]); // 1 second
        assert!((buffer.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vad_silent_audio() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let silent = create_silent_samples(1.0);
        assert!(!vad.contains_speech(&silent));
    }

    #[test]
    fn test_vad_speech_audio() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let speech = create_test_samples(1.0, 440.0, 0.5);
        assert!(vad.contains_speech(&speech));
    }

    #[test]
    fn test_vad_detect_segments() {
        let vad = VoiceActivityDetector::new(0.01, 16000);

        // Create audio: silence, speech, silence
        let mut samples = create_silent_samples(0.5);
        samples.extend(create_test_samples(1.0, 440.0, 0.5));
        samples.extend(create_silent_samples(0.5));

        let segments = vad.detect_speech_segments(&samples);
        assert!(!segments.is_empty());

        // The speech should be detected roughly in the middle
        let (start, end) = segments[0];
        assert!(start > 0);
        assert!(end <= samples.len());
        assert!(start < end);
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_duration_secs, 30);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let rms = VoiceActivityDetector::calculate_rms(&samples);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rms_empty() {
        let rms = VoiceActivityDetector::calculate_rms(&[]);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_chunk_buffer_empty() {
        let buffer = ChunkBuffer::new(0, AudioSource::Microphone, 0);
        assert!(!buffer.has_audio());
        assert!((buffer.duration_secs() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_chunk_buffer_take_samples() {
        let mut buffer = ChunkBuffer::new(0, AudioSource::Microphone, 0);
        buffer.add_samples(&[0.1, 0.2, 0.3]);
        assert!(buffer.has_audio());

        let samples = buffer.take_samples();
        assert_eq!(samples.len(), 3);
        assert!(!buffer.has_audio());
    }

    #[test]
    fn test_chunk_buffer_multiple_adds() {
        let mut buffer = ChunkBuffer::new(0, AudioSource::Microphone, 0);
        buffer.add_samples(&vec![0.0; 8000]); // 0.5 seconds
        buffer.add_samples(&vec![0.0; 8000]); // 0.5 seconds
        assert!((buffer.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_chunk_buffer_elapsed() {
        let buffer = ChunkBuffer::new(0, AudioSource::Microphone, 0);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(buffer.elapsed() >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_vad_empty_samples() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        assert!(!vad.contains_speech(&[]));
    }

    #[test]
    fn test_vad_detect_segments_empty() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let segments = vad.detect_speech_segments(&[]);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_vad_detect_segments_all_silence() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let silent = create_silent_samples(2.0);
        let segments = vad.detect_speech_segments(&silent);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_vad_detect_segments_all_speech() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let speech = create_test_samples(1.0, 440.0, 0.5);
        let segments = vad.detect_speech_segments(&speech);
        assert!(!segments.is_empty());
        // Speech covers the entire buffer
        let (start, _end) = segments[0];
        assert_eq!(start, 0);
    }

    #[test]
    fn test_vad_threshold_boundary() {
        // Amplitude exactly at threshold should not be detected as speech
        // since RMS of a sine wave with amplitude A is A / sqrt(2)
        let threshold = 0.5;
        let vad = VoiceActivityDetector::new(threshold, 16000);

        // Very quiet audio (RMS below threshold)
        let quiet = create_test_samples(1.0, 440.0, 0.001);
        assert!(!vad.contains_speech(&quiet));

        // Loud audio (RMS above threshold)
        let loud = create_test_samples(1.0, 440.0, 1.0);
        assert!(vad.contains_speech(&loud));
    }

    #[test]
    fn test_vad_zero_sample_rate_no_panic() {
        // Edge case: zero sample rate should not panic
        let vad = VoiceActivityDetector::new(0.01, 0);
        assert!(!vad.contains_speech(&[0.5, 0.5, 0.5]));
        assert!(vad.detect_speech_segments(&[0.5, 0.5]).is_empty());
    }

    #[test]
    fn test_chunk_config_custom() {
        let config = ChunkConfig {
            chunk_duration_secs: 60,
            vad_threshold: 0.05,
            sample_rate: 48000,
            min_chunk_duration_secs: 1.0,
        };
        assert_eq!(config.chunk_duration_secs, 60);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_rms_uniform_value() {
        let samples = vec![0.3; 100];
        let rms = VoiceActivityDetector::calculate_rms(&samples);
        assert!((rms - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_rms_single_sample() {
        let rms = VoiceActivityDetector::calculate_rms(&[0.7]);
        assert!((rms - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_process_chunk_merges_short_pause_into_single_transcription_turn() {
        let transcriber = Arc::new(MockTranscriber::default());
        let transcriber_dyn: Arc<dyn Transcriber> = transcriber.clone();
        let mut processor = ChunkProcessor::new_with_segment_id(
            ChunkConfig {
                min_chunk_duration_secs: 0.2,
                ..Default::default()
            },
            transcriber_dyn,
            41,
        );

        let mut samples = create_silent_samples(0.3);
        samples.extend(create_test_samples(0.6, 440.0, 0.5));
        samples.extend(create_silent_samples(0.25));
        samples.extend(create_test_samples(0.6, 660.0, 0.5));
        samples.extend(create_silent_samples(0.3));

        let mut buffer = processor.new_buffer(7, AudioSource::Loopback, 10_000);
        buffer.add_samples(&samples);

        let result = processor.process_chunk(buffer).unwrap();

        assert_eq!(result.chunk_id, 7);
        assert_eq!(result.chunk_start_ms, 10_000);
        assert_eq!(result.samples.len(), samples.len());
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].id, 41);
        assert_eq!(result.segments[0].chunk_id, 7);
        assert_eq!(result.segments[0].source, AudioSource::Loopback);
        assert_eq!(result.segments[0].text, "segment 1");
        assert!(result.segments[0].start_ms < result.segments[0].end_ms);
        assert_eq!(processor.next_segment_id(), 42);
        assert_eq!(transcriber.observed_calls().len(), 1);
    }

    #[test]
    fn test_process_chunk_splits_on_long_pause() {
        let transcriber = Arc::new(MockTranscriber::default());
        let transcriber_dyn: Arc<dyn Transcriber> = transcriber.clone();
        let mut processor = ChunkProcessor::new_with_segment_id(
            ChunkConfig {
                min_chunk_duration_secs: 0.2,
                ..Default::default()
            },
            transcriber_dyn,
            100,
        );

        let mut samples = create_silent_samples(0.3);
        samples.extend(create_test_samples(0.6, 440.0, 0.5));
        samples.extend(create_silent_samples(1.1));
        samples.extend(create_test_samples(0.6, 660.0, 0.5));
        samples.extend(create_silent_samples(0.3));

        let mut buffer = processor.new_buffer(8, AudioSource::Loopback, 5_000);
        buffer.add_samples(&samples);

        let result = processor.process_chunk(buffer).unwrap();

        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].id, 100);
        assert_eq!(result.segments[1].id, 101);
        assert!(result.segments[0].end_ms <= result.segments[1].start_ms);
        assert_eq!(processor.next_segment_id(), 102);
        assert_eq!(transcriber.observed_calls().len(), 2);
    }

    #[test]
    fn test_contains_speech_detects_brief_turn_in_long_chunk() {
        let vad = VoiceActivityDetector::new(0.01, 16000);
        let mut samples = create_silent_samples(4.0);
        samples.extend(create_test_samples(0.3, 440.0, 0.5));
        samples.extend(create_silent_samples(4.0));

        assert!(vad.contains_speech(&samples));
    }
}
