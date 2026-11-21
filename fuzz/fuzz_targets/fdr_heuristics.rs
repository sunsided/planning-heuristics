//! Fuzzes the heuristics on tasks parsed from Fast Downward `output.sas` text.
//!
//! The input is the file text, optionally followed by a NUL byte and the walk choices for
//! [`check_task`]. Random bytes rarely parse, so the corpus is seeded with the test fixtures
//! (`fuzz/corpus/fdr_heuristics/`, committed) to keep mutations close to well-formed files.

#![no_main]

use libfuzzer_sys::fuzz_target;
use planning_heuristics_fuzz::check_task;
use sas_plus::fdr::SasFile;

fuzz_target!(|data: &[u8]| {
    let (text, choices) = data
        .iter()
        .position(|&b| b == 0)
        .map_or((data, &[][..]), |nul| (&data[..nul], &data[nul + 1..]));
    let Ok(text) = core::str::from_utf8(text) else {
        return;
    };
    let Ok(file) = SasFile::parse(text) else {
        return;
    };

    check_task(&file.to_task(), choices);
});
