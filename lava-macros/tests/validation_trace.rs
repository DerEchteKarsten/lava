//! Tests for `#[validation_trace]`: the callsite is recorded for the call and always released
use lava_macros::validation_trace;

/// The macro expands to `crate::state::CallsiteGuard`, so this mirrors `lava::state`.
mod state {
    use std::{cell::Cell, panic::Location};

    thread_local! {
        pub static CALLSITE: Cell<Option<Location<'static>>> = Cell::new(None);
    }

    pub struct CallsiteGuard {
        owns: bool,
    }

    impl CallsiteGuard {
        pub fn enter(location: &'static Location<'static>) -> Self {
            let owns = CALLSITE.get().is_none();
            if owns {
                CALLSITE.set(Some(*location));
            }
            Self { owns }
        }
    }

    impl Drop for CallsiteGuard {
        fn drop(&mut self) {
            if self.owns {
                CALLSITE.set(None);
            }
        }
    }
}

/// Line of the location currently recorded, if any.
fn recorded_line() -> Option<u32> {
    state::CALLSITE.get().map(|location| location.line())
}

#[validation_trace]
fn returns_recorded_line() -> Option<u32> {
    recorded_line()
}

#[validation_trace]
fn without_tail_expression(seen: &mut Option<u32>) {
    *seen = recorded_line();
}

#[validation_trace]
fn early_return(early: bool) -> Option<u32> {
    if early {
        return recorded_line();
    }
    recorded_line()
}

#[validation_trace]
fn question_mark(fail: bool) -> Result<u32, String> {
    let value: u32 = if fail {
        Err("failed".to_owned())
    } else {
        Ok(1)
    }?;
    Ok(value)
}

#[validation_trace]
fn outer() -> (Option<u32>, Option<u32>, Option<u32>) {
    let before = recorded_line();
    let inner = returns_recorded_line();
    // The nested traced call must not clear the location of this one.
    let after = recorded_line();
    (before, inner, after)
}

#[validation_trace]
fn generic_with_args<T: Clone>(value: &T, times: usize) -> Vec<T> {
    vec![value.clone(); times]
}

struct Thing(u32);

impl Thing {
    #[validation_trace]
    fn method(&self) -> (u32, Option<u32>) {
        (self.0, recorded_line())
    }
}

#[validation_trace]
fn panics() {
    panic!("boom");
}

#[test]
fn records_the_callers_location_during_the_call() {
    let line = line!() + 1;
    let seen = returns_recorded_line();
    assert_eq!(seen, Some(line));
    assert_eq!(recorded_line(), None, "released after the call");
}

#[test]
fn works_without_a_tail_expression() {
    let mut seen = None;
    let line = line!() + 1;
    without_tail_expression(&mut seen);
    assert_eq!(seen, Some(line));
    assert_eq!(recorded_line(), None);
}

/// Regression: an early `return` used to skip the reset, leaving a stale location behind
/// that every later call on the thread then reported.
#[test]
fn early_return_releases_the_location() {
    assert!(early_return(true).is_some());
    assert_eq!(recorded_line(), None);

    let line = line!() + 1;
    assert_eq!(early_return(false), Some(line));
    assert_eq!(recorded_line(), None);
}

#[test]
fn question_mark_releases_the_location() {
    assert_eq!(question_mark(true), Err("failed".to_owned()));
    assert_eq!(recorded_line(), None);
    assert_eq!(question_mark(false), Ok(1));
    assert_eq!(recorded_line(), None);

    // A later call records its own location, not a stale one.
    let line = line!() + 1;
    assert_eq!(returns_recorded_line(), Some(line));
}

#[test]
fn nested_calls_report_the_outermost_caller() {
    let line = line!() + 1;
    let (before, inner, after) = outer();
    assert_eq!(before, Some(line));
    assert_eq!(
        inner,
        Some(line),
        "the user's call site, not a line inside `outer`"
    );
    assert_eq!(after, Some(line));
    assert_eq!(recorded_line(), None);
}

#[test]
fn arguments_generics_and_return_values_pass_through() {
    assert_eq!(generic_with_args(&"x", 3), ["x", "x", "x"]);
    let line = line!() + 1;
    assert_eq!(Thing(7).method(), (7, Some(line)));
    assert_eq!(recorded_line(), None);
}

#[test]
fn a_panic_releases_the_location() {
    assert!(std::panic::catch_unwind(panics).is_err());
    assert_eq!(recorded_line(), None);
}
