//! Half floats as generated bindings carry them: `F16`, the IEEE binary16 bits of a Slang `half`, converted from and to `f32` (rounded to nearest, ties to even)
use bytemuck::{Pod, Zeroable};

/// A Slang `half` (IEEE binary16 bits), in generated structs and push constants; vectors are
/// arrays of it. `F16::new(x)` rounds to nearest, ties to even (infinite beyond 65504).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct F16(pub u16);

impl F16 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(0x3C00);

    pub fn new(value: f32) -> Self {
        Self(f16_bits(value))
    }

    /// The exact `f32` of the half (NaNs stay NaNs).
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;
        let magnitude = match exponent {
            // Zero or subnormal: units of 2^-24.
            0 => {
                let value = mantissa as f32 * 2f32.powi(-24);
                return if sign != 0 { -value } else { value };
            }
            0x1F => 0x7F80_0000 | (mantissa << 13),
            _ => ((exponent + 127 - 15) << 23) | (mantissa << 13),
        };
        f32::from_bits(sign | magnitude)
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<F16> for f32 {
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

/// The binary16 bits of `value`, rounded to nearest, ties to even.
pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity, or a quiet NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |value: u32, shift: u32| {
        let (rest, halfway) = (value & ((1 << shift) - 1), 1 << (shift - 1));
        let truncated = value >> shift;
        truncated + (rest > halfway || (rest == halfway && truncated & 1 == 1)) as u32
    };
    if half_exponent <= 0 {
        // Subnormal (or zero): units of 2^-24.
        if half_exponent < -10 {
            return sign;
        }
        return sign | round(mantissa | 0x80_0000, (14 - half_exponent) as u32) as u16;
    }
    // A carry out of the mantissa raises the exponent, up to infinity, as it should.
    sign | round(((half_exponent as u32) << 23) | mantissa, 13) as u16
}
