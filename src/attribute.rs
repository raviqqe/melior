use crate::{
    context::{Context, ContextRef},
    utility::as_string_ref,
};
use mlir_sys::{mlirAttributeGetContext, mlirAttributeParseGet, MlirAttribute};
use std::marker::PhantomData;

pub struct Attribute<'c> {
    attribute: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Attribute<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            attribute: unsafe { mlirAttributeParseGet(context.to_raw(), as_string_ref(source)) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.attribute)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirAttribute {
        self.attribute
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            Attribute::parse(&Context::new(), attribute);
        }
    }

    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").context();
    }
}
