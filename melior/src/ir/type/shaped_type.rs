use super::TypeLike;
use mlir_sys::mlirShapedTypeGetRank;

/// Trait for shaped types.
pub trait ShapedType<'c>: TypeLike<'c> {
    /// Gets a rank.
    fn rank(&self) -> usize {
        (unsafe { mlirShapedTypeGetRank(self.to_raw()) }) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{r#type::MemRefType, Type},
        Context,
    };

    #[test]
    fn rank() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[], None, None).rank(),
            0
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0], None, None).rank(),
            1
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0, 0], None, None).rank(),
            2
        );
    }
}
