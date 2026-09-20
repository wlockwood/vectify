//! Background execution for tracing and auto-selection.
//!
//! Tracing a large image takes long enough that doing it on the UI thread would
//! make every slider feel broken. A single worker thread does the work, and the
//! request slot holds exactly one pending job: submitting a new one replaces
//! whatever was waiting. That gives coalescing for free, so dragging a slider
//! across twenty values does not queue twenty traces -- it runs the one in
//! flight, then jumps straight to the latest request.
//!
//! Results carry the generation they were computed for, and the UI discards
//! anything stale, so a slow trace finishing after the user has moved on can
//! never overwrite fresher output.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use vectify_core::auto::{auto_select, AutoConfig, AutoResult};
use vectify_core::color::LabCache;
use vectify_core::config::VectorizeConfig;
use vectify_core::model::VectorImage;
use vectify_core::raster::Raster;
use vectify_core::score::{score, render, ScoreConfig, ScoreReport};
use vectify_core::segment::scoring_reference;
use vectify_core::vectorize::{vectorize_with_cache, TraceStats};

pub enum Request {
    Trace {
        generation: u64,
        image: Arc<Raster>,
        config: Box<VectorizeConfig>,
        score: ScoreConfig,
    },
    Auto {
        generation: u64,
        image: Arc<Raster>,
        config: Box<AutoConfig>,
    },
}

impl Request {
    fn generation(&self) -> u64 {
        match self {
            Request::Trace { generation, .. } | Request::Auto { generation, .. } => *generation,
        }
    }
}

pub struct TraceOutcome {
    pub vector: VectorImage,
    pub stats: TraceStats,
    pub report: ScoreReport,
    pub rendered: Raster,
    pub config: VectorizeConfig,
}

pub enum Response {
    Traced {
        generation: u64,
        outcome: Box<TraceOutcome>,
    },
    AutoDone {
        generation: u64,
        result: Box<AutoResult>,
    },
    Failed {
        generation: u64,
        message: String,
    },
}

struct Slot {
    pending: Mutex<Option<Request>>,
    ready: Condvar,
}

pub struct Worker {
    slot: Arc<Slot>,
    rx: Receiver<Response>,
    generation: AtomicU64,
    /// (completed, total) for the auto-picker's progress bar.
    pub progress: Arc<(AtomicUsize, AtomicUsize)>,
    pub busy: Arc<AtomicUsize>,
}

impl Worker {
    pub fn new(repaint: impl Fn() + Send + Sync + 'static) -> Self {
        let slot = Arc::new(Slot {
            pending: Mutex::new(None),
            ready: Condvar::new(),
        });
        let (tx, rx) = channel();
        let progress = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
        let busy = Arc::new(AtomicUsize::new(0));

        {
            let slot = slot.clone();
            let progress = progress.clone();
            let busy = busy.clone();
            thread::Builder::new()
                .name("vectify-worker".into())
                .spawn(move || worker_loop(slot, tx, progress, busy, repaint))
                .expect("spawn worker thread");
        }

        Worker {
            slot,
            rx,
            generation: AtomicU64::new(0),
            progress,
            busy,
        }
    }

    /// Queue a job, replacing any job still waiting to start. Returns the
    /// generation number the caller should expect back.
    pub fn submit(&self, make: impl FnOnce(u64) -> Request) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let request = make(generation);
        let mut pending = self.slot.pending.lock().unwrap();
        *pending = Some(request);
        self.slot.ready.notify_one();
        generation
    }

    pub fn try_recv(&self) -> Option<Response> {
        self.rx.try_recv().ok()
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed) > 0
            || self.slot.pending.lock().map(|p| p.is_some()).unwrap_or(false)
    }
}

fn worker_loop(
    slot: Arc<Slot>,
    tx: Sender<Response>,
    progress: Arc<(AtomicUsize, AtomicUsize)>,
    busy: Arc<AtomicUsize>,
    repaint: impl Fn() + Send + Sync + 'static,
) {
    // Built once for the life of the thread: it is a 32k-entry table, and only
    // colour-keyed traces need it, but those may run on every slider tick.
    let cache = LabCache::new();
    loop {
        let request = {
            let mut pending = slot.pending.lock().unwrap();
            while pending.is_none() {
                pending = slot.ready.wait(pending).unwrap();
            }
            pending.take().unwrap()
        };

        busy.store(1, Ordering::Relaxed);
        let generation = request.generation();
        let response = run(request, &progress, &repaint, &cache);
        busy.store(0, Ordering::Relaxed);

        let failed = matches!(response, Response::Failed { .. });
        if tx.send(response).is_err() {
            return; // UI is gone
        }
        let _ = (generation, failed);
        repaint();
    }
}

fn run(
    request: Request,
    progress: &Arc<(AtomicUsize, AtomicUsize)>,
    repaint: &(impl Fn() + Send + Sync + 'static),
    cache: &LabCache,
) -> Response {
    match request {
        Request::Trace {
            generation,
            image,
            config,
            score: score_cfg,
        } => {
            let traced = vectorize_with_cache(&image, &config, cache);
            // Keyed-out colours are absent from the output on purpose; score
            // against an image with them removed so they do not count as error.
            let reference = scoring_reference(&image, &config.segment, cache);
            match score(&reference, &traced.image, &score_cfg) {
                Ok(report) => match render(&traced.image, 1) {
                    Ok(rendered) => Response::Traced {
                        generation,
                        outcome: Box::new(TraceOutcome {
                            vector: traced.image,
                            stats: traced.stats,
                            report,
                            rendered,
                            config: *config,
                        }),
                    },
                    Err(e) => Response::Failed {
                        generation,
                        message: format!("rendering the preview failed: {e}"),
                    },
                },
                Err(e) => Response::Failed {
                    generation,
                    message: format!("scoring failed: {e}"),
                },
            }
        }
        Request::Auto {
            generation,
            image,
            config,
        } => {
            progress.0.store(0, Ordering::Relaxed);
            progress.1.store(1, Ordering::Relaxed);
            let result = auto_select(&image, &config, &|done, total| {
                progress.0.store(done, Ordering::Relaxed);
                progress.1.store(total, Ordering::Relaxed);
                repaint();
            });
            Response::AutoDone {
                generation,
                result: Box::new(result),
            }
        }
    }
}
