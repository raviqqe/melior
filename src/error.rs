use std::{
    error,
    fmt::{self, Display, Formatter},
    str::Utf8Error,
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    BlockArgumentExpected(String),
    BlockArgumentPosition(String, usize),
    FunctionExpected(String),
    FunctionInputPosition(String, usize),
    FunctionResultPosition(String, usize),
    InvokeFunction,
    MemRefExpected(String),
    OperationResultExpected(String),
    OperationResultPosition(String, usize),
    ParsePassPipeline,
    RunPass,
    TupleExpected(String),
    TupleFieldPosition(String, usize),
    UnregisteredOperation(String),
    Utf8(Utf8Error),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {}", value)
            }
            Self::BlockArgumentPosition(block, position) => {
                write!(
                    formatter,
                    "block argument position {} out of range: {}",
                    position, block
                )
            }
            Self::FunctionExpected(r#type) => write!(formatter, "function expected: {}", r#type),
            Self::FunctionInputPosition(r#type, position) => write!(
                formatter,
                "function input position {} out of range: {}",
                position, r#type
            ),
            Self::FunctionResultPosition(r#type, position) => write!(
                formatter,
                "function result position {} out of range: {}",
                position, r#type
            ),
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::MemRefExpected(r#type) => write!(formatter, "mem-ref expected: {}", r#type),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {}", value)
            }
            Self::OperationResultPosition(operation, position) => {
                write!(
                    formatter,
                    "operation result position {} out of range: {}",
                    position, operation
                )
            }
            Self::ParsePassPipeline => write!(formatter, "failed to parse pass pipeline"),
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TupleExpected(r#type) => write!(formatter, "tuple expected: {}", r#type),
            Self::TupleFieldPosition(r#type, position) => {
                write!(
                    formatter,
                    "tuple field position {} out of range: {}",
                    position, r#type
                )
            }
            Self::UnregisteredOperation(name) => {
                write!(formatter, "unregistered operation: {}", name)
            }
            Self::Utf8(error) => error.fmt(formatter),
        }
    }
}

impl error::Error for Error {}

impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Error::Utf8(error)
    }
}
