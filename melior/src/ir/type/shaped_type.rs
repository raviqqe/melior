use super::{Type, TypeLike};
use mlir_sys::{mlirShapedTypeGetElementType, mlirShapedTypeGetRank, mlirShapedTypeHasRank};

/// Trait for shaped types.
pub trait ShapedType<'c>: TypeLike<'c> {
    /// Gets a element type.
    fn element(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirShapedTypeGetElementType(self.to_raw())) }
    }

    /// Gets a rank.
    fn rank(&self) -> usize {
        (unsafe { mlirShapedTypeGetRank(self.to_raw()) }) as usize
    }

    /// Checks if a type has a rank.
    fn has_rank(&self) -> bool {
        unsafe { mlirShapedTypeHasRank(self.to_raw()) }
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
    fn element() {
        let context = Context::new();
        let element_type = Type::index(&context);

        assert_eq!(
            MemRefType::new(element_type, &[], None, None).element(),
            element_type
        );
    }

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

    #[test]
    fn has_rank() {
        let context = Context::new();
        let element_type = Type::index(&context);

        assert_eq!(
            MemRefType::new(element_type, &[], None, None).has_rank(),
            true
        );
        assert_eq!(
            MemRefType::new(element_type, &[0], None, None).has_rank(),
            true,
        );
        assert_eq!(
            MemRefType::new(element_type, &[0, 0], None, None).has_rank(),
            true,
        );
    }
}
