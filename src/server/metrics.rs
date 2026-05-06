//! Minimal Prometheus text-exposition registry.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::RwLock;

/// Default histogram bucket bounds (seconds-scaled).
pub const DEFAULT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Sorted label set keyed by ASCII name.
pub type Labels = Vec<(String, String)>;

fn sort_labels(mut labels: Labels) -> Labels {
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    labels
}

fn format_labels(labels: &Labels) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let mut out = String::from("{");
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(k);
        out.push_str("=\"");
        for ch in v.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push('}');
    out
}

#[derive(Debug, Default)]
struct CounterFamily {
    help: String,
    values: HashMap<Labels, u64>,
}

#[derive(Debug, Default)]
struct HistogramFamily {
    help: String,
    buckets: Vec<f64>,
    series: HashMap<Labels, HistogramSeries>,
}

#[derive(Debug, Default, Clone)]
struct HistogramSeries {
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

/// Prometheus-compatible registry used by the server.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: RwLock<HashMap<String, CounterFamily>>,
    histograms: RwLock<HashMap<String, HistogramFamily>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_counter(&self, name: &str, help: &str) {
        let mut map = self.counters.write().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_string()).or_default().help = help.to_string();
    }

    pub fn register_histogram(&self, name: &str, help: &str, buckets: &[f64]) {
        let mut normalised: Vec<f64> = buckets.to_vec();
        normalised.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        normalised.dedup();
        let mut map = self.histograms.write().unwrap_or_else(|e| e.into_inner());
        let family = map.entry(name.to_string()).or_default();
        family.help = help.to_string();
        family.buckets = normalised;
    }

    pub fn counter_inc(&self, name: &str, labels: Labels, delta: u64) {
        let labels = sort_labels(labels);
        let mut map = self.counters.write().unwrap_or_else(|e| e.into_inner());
        let family = map.entry(name.to_string()).or_default();
        *family.values.entry(labels).or_insert(0) += delta;
    }

    pub fn histogram_record(&self, name: &str, labels: Labels, value: f64) {
        let labels = sort_labels(labels);
        let mut map = self.histograms.write().unwrap_or_else(|e| e.into_inner());
        let family = map.entry(name.to_string()).or_default();
        if family.buckets.is_empty() {
            family.buckets = DEFAULT_BUCKETS.to_vec();
        }
        let series = family
            .series
            .entry(labels)
            .or_insert_with(|| HistogramSeries {
                counts: vec![0; family.buckets.len()],
                sum: 0.0,
                count: 0,
            });
        if series.counts.len() < family.buckets.len() {
            series.counts.resize(family.buckets.len(), 0);
        }
        for (i, &upper) in family.buckets.iter().enumerate() {
            if value <= upper {
                series.counts[i] += 1;
            }
        }
        series.sum += value;
        series.count += 1;
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();

        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());
        let mut names: Vec<&String> = counters.keys().collect();
        names.sort();
        for name in names {
            let family = &counters[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} counter");
            let mut label_keys: Vec<&Labels> = family.values.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let _ = writeln!(out, "{name}{} {}", format_labels(labels), family.values[labels]);
            }
            out.push('\n');
        }
        drop(counters);

        let histograms = self.histograms.read().unwrap_or_else(|e| e.into_inner());
        let mut names: Vec<&String> = histograms.keys().collect();
        names.sort();
        for name in names {
            let family = &histograms[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} histogram");
            let mut label_keys: Vec<&Labels> = family.series.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let series = &family.series[labels];
                let base = format_labels(labels);
                let inner = trim_outer_braces(&base);
                let le_prefix: &str = if inner.is_empty() { "" } else { "," };
                for (i, &upper) in family.buckets.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "{name}_bucket{{{inner}{le_prefix}le=\"{}\"}} {}",
                        fmt_f64_prom(upper),
                        series.counts[i],
                    );
                }
                let _ = writeln!(
                    out,
                    "{name}_bucket{{{inner}{le_prefix}le=\"+Inf\"}} {}",
                    series.count
                );
                let _ = writeln!(out, "{name}_sum{} {}", base, fmt_f64_prom(series.sum));
                let _ = writeln!(out, "{name}_count{} {}", base, series.count);
            }
            out.push('\n');
        }

        out
    }
}

fn trim_outer_braces(formatted: &str) -> &str {
    if formatted.is_empty() {
        return "";
    }
    let inner = formatted
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(formatted);
    if inner.is_empty() { "" } else { inner }
}

fn fmt_f64_prom(v: f64) -> String {
    if v.is_infinite() {
        return if v.is_sign_positive() { "+Inf".into() } else { "-Inf".into() };
    }
    if v.is_nan() {
        return "NaN".into();
    }
    format!("{v}")
}
