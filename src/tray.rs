//! System tray icon for voxtype
//!
//! Shows daemon state (idle/recording/transcribing) and post-processing toggle
//! in the system tray via the StatusNotifierItem D-Bus protocol.
//!
//! Works with Cinnamon, KDE, GNOME (with AppIndicator extension), and other
//! freedesktop-compliant desktop environments.

use ksni::TrayMethods;

/// Tray icon state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
}

/// Messages sent from tray menu actions to the daemon
#[derive(Debug, Clone)]
pub enum TrayAction {
    TogglePostProcessing,
    Quit,
}

/// The tray icon implementation
#[derive(Debug)]
struct VoxtypeTray {
    state: TrayState,
    post_processing_enabled: bool,
    action_tx: tokio::sync::mpsc::UnboundedSender<TrayAction>,
}

impl ksni::Tray for VoxtypeTray {
    fn id(&self) -> String {
        "voxtype".into()
    }

    fn category(&self) -> ksni::Category {
        ksni::Category::ApplicationStatus
    }

    fn icon_name(&self) -> String {
        match self.state {
            TrayState::Idle => "audio-input-microphone".into(),
            TrayState::Recording => "media-record".into(),
            TrayState::Transcribing => "document-edit".into(),
        }
    }

    fn title(&self) -> String {
        let state_str = match self.state {
            TrayState::Idle => "Idle",
            TrayState::Recording => "Recording...",
            TrayState::Transcribing => "Transcribing...",
        };
        let pp = if self.post_processing_enabled {
            " [PP]"
        } else {
            ""
        };
        format!("Voxtype: {}{}", state_str, pp)
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        let description = match self.state {
            TrayState::Idle => "Ready - hold Scroll Lock to record",
            TrayState::Recording => "Recording audio...",
            TrayState::Transcribing => "Transcribing speech...",
        };
        ksni::ToolTip {
            title: self.title(),
            description: description.into(),
            ..Default::default()
        }
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;

        let state_label = match self.state {
            TrayState::Idle => "Idle",
            TrayState::Recording => "Recording...",
            TrayState::Transcribing => "Transcribing...",
        };

        vec![
            StandardItem {
                label: state_label.into(),
                enabled: false,
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            CheckmarkItem {
                label: "Post-Processing".into(),
                checked: self.post_processing_enabled,
                activate: Box::new(|this: &mut Self| {
                    this.post_processing_enabled = !this.post_processing_enabled;
                    let _ = this
                        .action_tx
                        .send(TrayAction::TogglePostProcessing);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(|this: &mut Self| {
                    let _ = this.action_tx.send(TrayAction::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

/// Handle to the running tray icon, used to update its state
pub struct TrayHandle {
    handle: ksni::Handle<VoxtypeTray>,
}

impl TrayHandle {
    /// Update the displayed state (idle/recording/transcribing)
    pub async fn set_state(&self, state: TrayState) {
        self.handle
            .update(move |tray| {
                tray.state = state;
            })
            .await;
    }

    /// Update the post-processing enabled status
    pub async fn set_post_processing(&self, enabled: bool) {
        self.handle
            .update(move |tray| {
                tray.post_processing_enabled = enabled;
            })
            .await;
    }

    /// Create a clone of the handle for the state watcher task
    pub fn clone_for_watcher(&self) -> TrayHandle {
        TrayHandle {
            handle: self.handle.clone(),
        }
    }
}

/// Spawn the system tray icon.
///
/// Returns a handle for updating the tray state, and a receiver for menu actions.
pub async fn spawn_tray(
    post_processing_enabled: bool,
) -> Option<(TrayHandle, tokio::sync::mpsc::UnboundedReceiver<TrayAction>)> {
    let (action_tx, action_rx) = tokio::sync::mpsc::unbounded_channel();

    let tray = VoxtypeTray {
        state: TrayState::Idle,
        post_processing_enabled,
        action_tx,
    };

    match tray.spawn().await {
        Ok(handle) => {
            tracing::info!("System tray icon started");
            Some((TrayHandle { handle }, action_rx))
        }
        Err(e) => {
            tracing::warn!("Failed to start system tray icon: {}", e);
            None
        }
    }
}
