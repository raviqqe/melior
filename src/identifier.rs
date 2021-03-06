use crate::{
    context::{Context, ContextRef},
    utility::as_string_ref,
};
use mlir_sys::{mlirIdentifierGet, mlirIdentifierGetContext, MlirIdentifier};
use std::marker::PhantomData;

pub struct Identifier<'c> {
    identifier: MlirIdentifier,
    _context: PhantomData<&'c Context>,
}

impl<'c> Identifier<'c> {
    pub fn new(context: &Context, name: &str) -> Self {
        Self {
            identifier: unsafe { mlirIdentifierGet(context.to_raw(), as_string_ref(name)) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirIdentifierGetContext(self.identifier)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirIdentifier {
        self.identifier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Identifier::new(&Context::new(), "foo");
    }

    #[test]
    fn context() {
        Identifier::new(&Context::new(), "foo").context();
    }
}
