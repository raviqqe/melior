use super::Pass;
use mlir_sys::{
    mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsPrintOpStats,
    mlirCreateTransformsSymbolPrivatize,
};

/// Creates a pass to canonicalize IR.
pub fn canonicalizer() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCanonicalizer)
}

/// Creates a pass to apply CSE.
pub fn cse() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCSE)
}

/// Creates a pass to privatize symbols.
pub fn symbol_privatize() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSymbolPrivatize)
}

/// Creates a pass to print operation statistics.
pub fn print_operation_stats() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsPrintOpStats)
}
