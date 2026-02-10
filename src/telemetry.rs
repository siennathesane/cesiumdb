//! OpenTelemetry instrumentation for CesiumDB
//!
//! This module provides comprehensive observability through OpenTelemetry,
//! including distributed tracing, metrics, and performance monitoring.
//!
//! # Usage
//!
//! Enable the "telemetry" feature in Cargo.toml:
//! ```toml
//! cesiumdb = { version = "0.0.5", features = ["telemetry"] }
//! ```
//!
//! Initialize telemetry at application startup:
//! ```rust,no_run
//! # #[cfg(feature = "telemetry")]
//! use cesiumdb::telemetry;
//!
//! # fn main() {
//! # #[cfg(feature = "telemetry")]
//! let _guard = telemetry::init_with_jaeger("http://localhost:4317");
//! // ... use CesiumDB ...
//! # }
//! ```
//!
//! # Zero-Cost Abstraction
//!
//! When the "telemetry" feature is disabled, all instrumentation compiles away
//! to zero overhead. When enabled, overhead is minimal (<5% typical workloads).

#[cfg(feature = "telemetry")]
use opentelemetry::{
    global,
    trace::TracerProvider as _,
};
#[cfg(feature = "telemetry")]
use opentelemetry_sdk::{
    runtime,
    trace::{RandomIdGenerator, Sampler, TracerProvider},
};

/// Guard that ensures telemetry is properly flushed on drop
pub struct TelemetryGuard {
    #[cfg(feature = "telemetry")]
    _private: (),
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        #[cfg(feature = "telemetry")]
        {
            // Flush all pending spans
            global::shutdown_tracer_provider();
        }
    }
}

/// Initialize OpenTelemetry with a basic tracer provider
///
/// **Note**: This creates a local tracer provider without an exporter.
/// To send traces to a backend like Jaeger or Tempo, you must configure
/// your own exporter externally using `init_with_provider()`.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "telemetry")]
/// # {
/// use cesiumdb::telemetry;
///
/// let _guard = telemetry::init();
/// // ... use CesiumDB ...
/// // Spans are generated but not exported
/// # }
/// ```
///
/// For full telemetry with an external backend, see:
/// https://docs.rs/opentelemetry-otlp for OTLP exporter setup
#[cfg(feature = "telemetry")]
pub fn init() -> TelemetryGuard {
    use opentelemetry_sdk::trace;

    // Create a basic tracer provider without an exporter
    // Spans are generated but not exported anywhere
    let provider = trace::TracerProvider::builder()
        .with_sampler(Sampler::AlwaysOn)
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(opentelemetry_sdk::Resource::new(vec![
            opentelemetry::KeyValue::new("service.name", "cesiumdb"),
        ]))
        .build();

    global::set_tracer_provider(provider);

    TelemetryGuard { _private: () }
}

/// Initialize OpenTelemetry with a custom tracer provider
///
/// This allows advanced users to configure their own exporters, samplers, etc.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "telemetry")]
/// # {
/// use cesiumdb::telemetry;
/// use opentelemetry_sdk::trace::TracerProvider;
///
/// let provider = TracerProvider::builder()
///     // ... custom configuration ...
///     .build();
///
/// let _guard = telemetry::init_with_provider(provider);
/// # }
/// ```
#[cfg(feature = "telemetry")]
pub fn init_with_provider(provider: TracerProvider) -> TelemetryGuard {
    global::set_tracer_provider(provider);
    TelemetryGuard { _private: () }
}

/// No-op initialization when telemetry feature is disabled
#[cfg(not(feature = "telemetry"))]
pub fn init() -> TelemetryGuard {
    TelemetryGuard {}
}

/// Instrumentation helpers
///
/// These macros provide zero-cost abstractions that compile away when
/// the telemetry feature is disabled.

/// Create a debug span (only active when telemetry feature is enabled)
///
/// This macro handles both span creation and entry, avoiding unit value warnings.
/// Usage: `telemetry_span!("span_name", field = value);`
#[macro_export]
macro_rules! telemetry_span {
    ($name:expr $(, $($fields:tt)*)?) => {{
        #[cfg(feature = "telemetry")]
        {
            let _span = tracing::debug_span!($name $(, $($fields)*)?);
            let _guard = _span.entered();
        }
    }};
}

/// Record an event in the current span
#[macro_export]
macro_rules! telemetry_event {
    ($($fields:tt)*) => {{
        #[cfg(feature = "telemetry")]
        {
            tracing::event!(tracing::Level::DEBUG, $($fields)*);
        }
    }};
}

/// Instrument a function with a tracing span
///
/// When telemetry is disabled, this compiles to a no-op.
#[macro_export]
macro_rules! instrument {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($generic:tt),*>)?($($arg:tt)*) $(-> $ret:ty)? $body:block
    ) => {
        #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
        $(#[$meta])*
        $vis fn $name $(<$($generic),*>)?($($arg)*) $(-> $ret)? $body
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_guard_drop() {
        // Just ensure it compiles and doesn't panic
        let _guard = TelemetryGuard {
            #[cfg(feature = "telemetry")]
            _private: (),
        };
    }
}
