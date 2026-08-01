//! Meeting transcription mode
//!
//! Provides continuous meeting transcription with chunked processing,
//! speaker attribution, and export capabilities.
//!
//! Enables transcription of longer meetings (up to 3 hours) with
//! automatic chunking and speaker separation.
//!
//! # Architecture
//!
//! ```text
//! Mic + Loopback → ChunkProcessor → VAD → Transcription → Storage
//!                                           ↓
//!                                   Diarization (Phase 3)
//! ```
//!
//! # Phases
//!
//! - **Phase 1 (v0.5.0):** Basic meeting mode with chunked processing
//! - **Phase 2 (v0.5.1):** Dual audio + simple You/Remote attribution
//! - **Phase 3 (v0.5.2):** ML-based speaker diarization
//! - **Phase 4 (v0.6.0):** Remote server sync for corporate deployments
//! - **Phase 5 (v0.6.1):** AI summarization with action items

pub mod chunk;
pub mod data;
pub mod diarization;
pub mod export;
pub mod state;
pub mod storage;
pub mod summary;

pub use chunk::{ChunkBuffer, ChunkConfig, ChunkProcessor, ProcessedChunk, VoiceActivityDetector};
pub use data::{
    ActionItem, AudioSource, MeetingData, MeetingId, MeetingMetadata, MeetingStatus,
    MeetingSummary, Transcript, TranscriptSegment,
};
pub use export::{export_meeting, export_meeting_to_file, ExportFormat, ExportOptions};
pub use state::{ChunkState, MeetingState};
pub use storage::{MeetingStorage, StorageConfig, StorageError};

use crate::error::{MeetingError, Result};
use crate::meeting::diarization::Diarizer;
use crate::transcribe::{self, Transcriber};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Meeting daemon configuration
#[derive(Debug, Clone)]
pub struct MeetingConfig {
    /// Enable meeting mode
    pub enabled: bool,
    /// Duration of each audio chunk in seconds
    pub chunk_duration_secs: u32,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Whether to retain raw audio files
    pub retain_audio: bool,
    /// Maximum meeting duration in minutes (0 = unlimited)
    pub max_duration_mins: u32,
    /// Speaker diarization configuration
    pub diarization: diarization::DiarizationConfig,
}

impl Default for MeetingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chunk_duration_secs: 30,
            storage: StorageConfig::default(),
            retain_audio: false,
            max_duration_mins: 180,
            diarization: diarization::DiarizationConfig::default(),
        }
    }
}

/// Events from the meeting daemon
#[derive(Debug)]
pub enum MeetingEvent {
    /// Meeting started
    Started { meeting_id: MeetingId },
    /// Chunk processed
    ChunkProcessed {
        chunk_id: u32,
        segments: Vec<TranscriptSegment>,
    },
    /// Meeting paused
    Paused,
    /// Meeting resumed
    Resumed,
    /// Meeting stopped
    Stopped { meeting_id: MeetingId },
    /// Error occurred
    Error(String),
}

/// Meeting daemon for continuous transcription
pub struct MeetingDaemon {
    config: MeetingConfig,
    state: MeetingState,
    storage: MeetingStorage,
    current_meeting: Option<MeetingData>,
    transcriber: Option<Arc<dyn Transcriber>>,
    diarizer: Box<dyn Diarizer>,
    engine_name: String,
    next_segment_id: u32,
    event_tx: mpsc::Sender<MeetingEvent>,
}

impl MeetingDaemon {
    /// Create a new meeting daemon
    pub fn new(
        config: MeetingConfig,
        app_config: &crate::config::Config,
        event_tx: mpsc::Sender<MeetingEvent>,
    ) -> Result<Self> {
        let storage = MeetingStorage::open(config.storage.clone())
            .map_err(|e| MeetingError::Storage(e.to_string()))?;

        let transcriber: Arc<dyn Transcriber> =
            Arc::from(transcribe::create_transcriber(app_config)?);
        let diarizer = diarization::create_diarizer(&config.diarization);
        let engine_name = format!("{:?}", app_config.engine).to_lowercase();

        Ok(Self {
            config,
            state: MeetingState::Idle,
            storage,
            current_meeting: None,
            transcriber: Some(transcriber),
            diarizer,
            engine_name,
            next_segment_id: 0,
            event_tx,
        })
    }

    /// Start a new meeting
    pub async fn start(&mut self, title: Option<String>) -> Result<MeetingId> {
        if !self.state.is_idle() {
            return Err(MeetingError::AlreadyInProgress.into());
        }

        self.next_segment_id = 0;
        self.diarizer.reset();

        // Create meeting
        let mut meeting = MeetingData::new(title);
        meeting.metadata.model = Some(self.engine_name.clone());

        // Create storage directory
        let storage_path = self
            .storage
            .create_meeting(&meeting.metadata)
            .map_err(|e| MeetingError::Storage(e.to_string()))?;
        meeting.metadata.storage_path = Some(storage_path);

        let meeting_id = meeting.metadata.id;
        self.current_meeting = Some(meeting);
        self.state = MeetingState::start();

        let _ = self
            .event_tx
            .send(MeetingEvent::Started { meeting_id })
            .await;
        tracing::info!("Meeting started: {}", meeting_id);

        Ok(meeting_id)
    }

    /// Pause the current meeting
    pub async fn pause(&mut self) -> Result<()> {
        if !self.state.is_active() {
            return Err(MeetingError::NotActive.into());
        }

        self.state = std::mem::take(&mut self.state).pause();
        let _ = self.event_tx.send(MeetingEvent::Paused).await;
        tracing::info!("Meeting paused");

        Ok(())
    }

    /// Resume a paused meeting
    pub async fn resume(&mut self) -> Result<()> {
        if !self.state.is_paused() {
            return Err(MeetingError::NotPaused.into());
        }

        self.state = std::mem::take(&mut self.state).resume();
        let _ = self.event_tx.send(MeetingEvent::Resumed).await;
        tracing::info!("Meeting resumed");

        Ok(())
    }

    /// Stop the current meeting
    pub async fn stop(&mut self) -> Result<MeetingId> {
        if self.state.is_idle() {
            return Err(MeetingError::NotInProgress.into());
        }

        self.state = std::mem::take(&mut self.state).stop();
        let meeting_id = self
            .current_meeting
            .as_ref()
            .map(|m| m.metadata.id)
            .unwrap_or_default();

        let finalize_result = if let Some(ref mut meeting) = self.current_meeting {
            meeting.complete();
            meeting.metadata.chunk_count = meeting.transcript.total_chunks;

            self.storage
                .save_transcript(&meeting.metadata.id, &meeting.transcript)
                .map_err(|e| MeetingError::Storage(e.to_string()))
                .and_then(|_| {
                    self.storage
                        .update_meeting(&meeting.metadata)
                        .map_err(|e| MeetingError::Storage(e.to_string()))
                })
        } else {
            Ok(())
        };

        match &finalize_result {
            Ok(()) => {
                let _ = self
                    .event_tx
                    .send(MeetingEvent::Stopped { meeting_id })
                    .await;
                tracing::info!("Meeting stopped: {}", meeting_id);
            }
            Err(err) => {
                let _ = self
                    .event_tx
                    .send(MeetingEvent::Error(format!(
                        "Failed to finalize meeting {}: {}",
                        meeting_id, err
                    )))
                    .await;
            }
        }

        // Clean up
        self.state = std::mem::take(&mut self.state).finalize();
        self.current_meeting = None;
        self.next_segment_id = 0;
        self.diarizer.reset();

        finalize_result.map_err(crate::error::VoxtypeError::from)?;
        Ok(meeting_id)
    }

    /// Persist the current meeting transcript and metadata to storage.
    pub fn persist_current_meeting(&self) -> Result<()> {
        let Some(meeting) = self.current_meeting.as_ref() else {
            return Ok(());
        };

        self.storage
            .save_transcript(&meeting.metadata.id, &meeting.transcript)
            .map_err(|e| MeetingError::Storage(e.to_string()))?;
        self.storage
            .update_meeting(&meeting.metadata)
            .map_err(|e| MeetingError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get current meeting state
    pub fn state(&self) -> &MeetingState {
        &self.state
    }

    /// Get current meeting ID if one is active
    pub fn current_meeting_id(&self) -> Option<MeetingId> {
        self.current_meeting.as_ref().map(|m| m.metadata.id)
    }

    /// Get mutable access to current meeting data (for dedup, etc.)
    pub fn current_meeting_mut(&mut self) -> Option<&mut MeetingData> {
        self.current_meeting.as_mut()
    }

    /// Process a chunk of audio
    pub async fn process_chunk(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<Option<Vec<TranscriptSegment>>> {
        self.process_chunk_with_source(samples, AudioSource::Microphone)
            .await
    }

    /// Process a chunk of audio with a specific source label
    pub async fn process_chunk_with_source(
        &mut self,
        samples: Vec<f32>,
        source: AudioSource,
    ) -> Result<Option<Vec<TranscriptSegment>>> {
        let chunk_duration_ms = samples.len() as u64 * 1000 / 16_000;
        let start_offset_ms = self
            .state
            .elapsed()
            .map(|duration| duration.as_millis() as u64)
            .or_else(|| {
                self.current_meeting
                    .as_ref()
                    .map(|meeting| meeting.transcript.duration_ms())
            })
            .unwrap_or(0)
            .saturating_sub(chunk_duration_ms);

        self.process_chunk_with_source_at(samples, source, start_offset_ms)
            .await
    }

    /// Process a chunk of audio with an explicit wall-clock start offset.
    pub async fn process_chunk_with_source_at(
        &mut self,
        samples: Vec<f32>,
        source: AudioSource,
        start_offset_ms: u64,
    ) -> Result<Option<Vec<TranscriptSegment>>> {
        if !self.state.is_active() {
            return Ok(None);
        }

        let Some(ref transcriber) = self.transcriber else {
            return Err(MeetingError::TranscriberNotInitialized.into());
        };

        let chunk_id = self.state.chunks_processed();
        let chunk_config = ChunkConfig {
            chunk_duration_secs: self.config.chunk_duration_secs,
            ..Default::default()
        };

        let mut processor = ChunkProcessor::new_with_segment_id(
            chunk_config,
            transcriber.clone(),
            self.next_segment_id,
        );
        let mut buffer = processor.new_buffer(chunk_id, source, start_offset_ms);
        buffer.add_samples(&samples);

        let result = processor
            .process_chunk(buffer)
            .map_err(crate::error::VoxtypeError::Transcribe)?;
        self.next_segment_id = processor.next_segment_id();

        let mut segments = result.segments;
        if self.config.diarization.enabled && !segments.is_empty() {
            let diarized =
                self.diarizer
                    .diarize(&result.samples, result.chunk_start_ms, source, &segments);

            if !diarized.is_empty() {
                let mut reusable_ids = segments.iter().map(|segment| segment.id);
                let mut refined_segments = Vec::with_capacity(diarized.len());

                for diarized_segment in diarized {
                    let text = diarized_segment.text.trim();
                    if text.is_empty() {
                        continue;
                    }

                    let segment_id = if let Some(existing) = reusable_ids.next() {
                        existing
                    } else {
                        let new_id = self.next_segment_id;
                        self.next_segment_id += 1;
                        new_id
                    };

                    let mut segment = TranscriptSegment::new(
                        segment_id,
                        diarized_segment.start_ms,
                        diarized_segment.end_ms,
                        text.to_string(),
                        chunk_id,
                    );
                    segment.source = source;
                    segment.speaker_id = Some(diarized_segment.speaker.display_name());
                    segment.confidence = Some(diarized_segment.confidence);
                    refined_segments.push(segment);
                }

                if !refined_segments.is_empty() {
                    segments = refined_segments;
                } else {
                    tracing::warn!(
                        "Diarizer '{}' returned only empty refined segments for chunk {}",
                        self.diarizer.name(),
                        chunk_id
                    );
                }
            }
        }

        // Add segments to transcript
        if let Some(ref mut meeting) = self.current_meeting {
            if !self.config.diarization.enabled && matches!(source, AudioSource::Loopback) {
                meeting.transcript.segments.extend(segments.iter().cloned());
                meeting.transcript.segments.sort_by_key(|segment| {
                    (
                        segment.start_ms,
                        segment.end_ms,
                        segment.chunk_id,
                        segment.id,
                    )
                });
            } else {
                meeting.transcript.add_segments(segments.iter().cloned());
            }
            meeting.transcript.total_chunks = chunk_id + 1;
            meeting.metadata.chunk_count = meeting.transcript.total_chunks;
        }

        if !segments.is_empty() {
            if let Err(err) = self.persist_current_meeting() {
                tracing::error!("Failed to persist active meeting state: {}", err);
                let _ = self
                    .event_tx
                    .send(MeetingEvent::Error(format!(
                        "Failed to persist active meeting transcript: {}",
                        err
                    )))
                    .await;
            }
        }

        // Advance state
        self.state = std::mem::take(&mut self.state).next_chunk();

        // Send event
        let _ = self
            .event_tx
            .send(MeetingEvent::ChunkProcessed {
                chunk_id,
                segments: segments.clone(),
            })
            .await;

        Ok(Some(segments))
    }

    /// Get storage access
    pub fn storage(&self) -> &MeetingStorage {
        &self.storage
    }
}

/// List meetings from storage
pub fn list_meetings(
    config: &MeetingConfig,
    limit: Option<u32>,
) -> std::result::Result<Vec<MeetingMetadata>, StorageError> {
    let storage = MeetingStorage::open(config.storage.clone())?;
    storage.list_meetings(limit)
}

/// Get a meeting by ID (or "latest")
pub fn get_meeting(
    config: &MeetingConfig,
    id_str: &str,
) -> std::result::Result<MeetingData, StorageError> {
    let storage = MeetingStorage::open(config.storage.clone())?;
    let id = storage.resolve_meeting_id(id_str)?;
    storage.load_meeting_data(&id)
}

/// Export a meeting
pub fn export_meeting_by_id(
    config: &MeetingConfig,
    id_str: &str,
    format: ExportFormat,
    options: &ExportOptions,
) -> std::result::Result<String, StorageError> {
    let meeting = get_meeting(config, id_str)?;
    export_meeting(&meeting, format, options)
        .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TranscribeError;
    use crate::meeting::diarization::{DiarizedSegment, SpeakerId};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    struct MockTranscriber {
        text: String,
        calls: Mutex<Vec<usize>>,
    }

    impl MockTranscriber {
        fn new(text: &str) -> Self {
            Self {
                text: text.to_string(),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl Transcriber for MockTranscriber {
        fn transcribe(&self, samples: &[f32]) -> std::result::Result<String, TranscribeError> {
            self.calls.lock().unwrap().push(samples.len());
            Ok(self.text.clone())
        }
    }

    struct ResetTrackingDiarizer {
        reset_calls: Arc<AtomicUsize>,
    }

    impl ResetTrackingDiarizer {
        fn new(reset_calls: Arc<AtomicUsize>) -> Self {
            Self { reset_calls }
        }
    }

    impl Diarizer for ResetTrackingDiarizer {
        fn diarize(
            &mut self,
            _samples: &[f32],
            _chunk_start_ms: u64,
            source: AudioSource,
            transcript_segments: &[TranscriptSegment],
        ) -> Vec<DiarizedSegment> {
            let speaker = match source {
                AudioSource::Microphone => SpeakerId::You,
                AudioSource::Loopback => SpeakerId::Remote,
                AudioSource::Unknown => SpeakerId::Unknown,
            };

            transcript_segments
                .iter()
                .map(|segment| DiarizedSegment {
                    speaker: speaker.clone(),
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    text: segment.text.clone(),
                    confidence: 1.0,
                })
                .collect()
        }

        fn name(&self) -> &'static str {
            "reset-tracking"
        }

        fn reset(&mut self) {
            self.reset_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn create_test_meeting_daemon(
        diarization_enabled: bool,
        diarizer: Box<dyn Diarizer>,
    ) -> (MeetingDaemon, TempDir) {
        let temp = TempDir::new().expect("tempdir");
        let config = MeetingConfig {
            enabled: true,
            storage: StorageConfig {
                storage_path: temp.path().to_path_buf(),
                retain_audio: false,
                max_meetings: 0,
            },
            diarization: diarization::DiarizationConfig {
                enabled: diarization_enabled,
                ..Default::default()
            },
            ..Default::default()
        };
        let storage = MeetingStorage::open(config.storage.clone()).expect("storage");
        let (event_tx, _event_rx) = mpsc::channel(8);

        (
            MeetingDaemon {
                config,
                state: MeetingState::Idle,
                storage,
                current_meeting: None,
                transcriber: Some(Arc::new(MockTranscriber::new("hello world"))),
                diarizer,
                engine_name: "test".to_string(),
                next_segment_id: 0,
                event_tx,
            },
            temp,
        )
    }

    #[test]
    fn test_meeting_config_default() {
        let config = MeetingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.chunk_duration_secs, 30);
        assert_eq!(config.max_duration_mins, 180);
    }

    #[test]
    fn test_meeting_state_transitions() {
        let state = MeetingState::Idle;
        assert!(state.is_idle());

        let state = MeetingState::start();
        assert!(state.is_active());

        let state = state.pause();
        assert!(state.is_paused());

        let state = state.resume();
        assert!(state.is_active());

        let state = state.stop();
        assert!(state.is_finalizing());

        let state = state.finalize();
        assert!(state.is_idle());
    }

    #[tokio::test]
    async fn test_process_chunk_persists_transcript_during_active_meeting() {
        let (mut daemon, _temp) =
            create_test_meeting_daemon(false, Box::new(diarization::simple::SimpleDiarizer::new()));
        let meeting_id = daemon
            .start(Some("Persistence Test".to_string()))
            .await
            .expect("meeting start");

        let samples = vec![0.2; 16_000];
        let result = daemon
            .process_chunk_with_source_at(samples, AudioSource::Microphone, 0)
            .await
            .expect("process chunk");

        assert!(result.is_some());

        let persisted = daemon
            .storage()
            .load_transcript(&meeting_id)
            .expect("persisted transcript");
        assert_eq!(persisted.segments.len(), 1);
        assert_eq!(persisted.segments[0].text, "hello world");

        let metadata = daemon
            .storage()
            .get_meeting(&meeting_id)
            .expect("metadata lookup")
            .expect("meeting metadata");
        assert_eq!(metadata.chunk_count, 1);
    }

    #[tokio::test]
    async fn test_process_chunk_continues_when_incremental_persist_fails() {
        let (mut daemon, _temp) =
            create_test_meeting_daemon(false, Box::new(diarization::simple::SimpleDiarizer::new()));
        let meeting_id = daemon
            .start(Some("Persistence Failure".to_string()))
            .await
            .expect("meeting start");

        let storage_path = daemon
            .storage()
            .get_meeting_path(&meeting_id)
            .expect("meeting path");
        std::fs::remove_dir_all(&storage_path).expect("remove meeting directory");

        let samples = vec![0.2; 16_000];
        let result = daemon
            .process_chunk_with_source_at(samples, AudioSource::Microphone, 0)
            .await
            .expect("chunk processing should continue");

        assert!(result.is_some());
        assert_eq!(daemon.state().chunks_processed(), 1);
        assert_eq!(
            daemon
                .current_meeting
                .as_ref()
                .expect("in-memory meeting")
                .transcript
                .segments
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_process_chunk_with_source_uses_wall_clock_offset() {
        let (mut daemon, _temp) =
            create_test_meeting_daemon(false, Box::new(diarization::simple::SimpleDiarizer::new()));
        let meeting_id = daemon
            .start(Some("Wall Clock".to_string()))
            .await
            .expect("meeting start");

        {
            let meeting = daemon.current_meeting.as_mut().expect("meeting");
            meeting.transcript.add_segment(TranscriptSegment::new(
                99,
                0,
                1_000,
                "earlier turn".to_string(),
                0,
            ));
            meeting.transcript.total_chunks = 1;
            meeting.metadata.chunk_count = 1;
        }
        daemon
            .persist_current_meeting()
            .expect("persist seed transcript");

        daemon.state = MeetingState::Active {
            started_at: Instant::now() - Duration::from_secs(10),
            current_chunk: ChunkState::Recording {
                started_at: Instant::now(),
            },
            chunks_processed: 1,
        };

        let samples = vec![0.2; 16_000];
        let result = daemon
            .process_chunk_with_source(samples, AudioSource::Microphone)
            .await
            .expect("process chunk")
            .expect("segments");

        assert_eq!(result.len(), 1);
        assert!(result[0].start_ms >= 8_000);

        let persisted = daemon
            .storage()
            .load_transcript(&meeting_id)
            .expect("persisted transcript");
        assert!(
            persisted
                .segments
                .iter()
                .any(|segment| segment.start_ms >= 8_000),
            "expected a wall-clock timestamp in the persisted transcript"
        );
    }

    #[tokio::test]
    async fn test_stop_cleans_up_even_if_final_persist_fails() {
        let (mut daemon, _temp) =
            create_test_meeting_daemon(false, Box::new(diarization::simple::SimpleDiarizer::new()));
        let meeting_id = daemon
            .start(Some("Stop Failure".to_string()))
            .await
            .expect("meeting start");

        let storage_path = daemon
            .storage()
            .get_meeting_path(&meeting_id)
            .expect("meeting path");
        std::fs::remove_dir_all(&storage_path).expect("remove meeting directory");

        let err = daemon
            .stop()
            .await
            .expect_err("stop should report persist error");
        assert!(err.to_string().contains("storage"));
        assert!(daemon.state().is_idle());
        assert!(daemon.current_meeting.is_none());
        assert_eq!(daemon.next_segment_id, 0);
    }

    #[tokio::test]
    async fn test_meeting_diarizer_resets_on_start_and_stop() {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let (mut daemon, _temp) = create_test_meeting_daemon(
            false,
            Box::new(ResetTrackingDiarizer::new(reset_calls.clone())),
        );

        daemon
            .start(Some("First".to_string()))
            .await
            .expect("first start");
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);

        daemon.stop().await.expect("first stop");
        assert_eq!(reset_calls.load(Ordering::SeqCst), 2);

        daemon
            .start(Some("Second".to_string()))
            .await
            .expect("second start");
        assert_eq!(reset_calls.load(Ordering::SeqCst), 3);
    }
}
