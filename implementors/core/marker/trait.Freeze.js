(function() {var implementors = {};
implementors["melior"] = [{"text":"impl Freeze for <a class=\"struct\" href=\"melior/struct.Context.html\" title=\"struct melior::Context\">Context</a>","synthetic":true,"types":["melior::context::Context"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/struct.ContextRef.html\" title=\"struct melior::ContextRef\">ContextRef</a>&lt;'a&gt;","synthetic":true,"types":["melior::context::ContextRef"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/dialect/struct.Handle.html\" title=\"struct melior::dialect::Handle\">Handle</a>","synthetic":true,"types":["melior::dialect::handle::Handle"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/dialect/struct.Registry.html\" title=\"struct melior::dialect::Registry\">Registry</a>","synthetic":true,"types":["melior::dialect::registry::Registry"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/dialect/struct.Dialect.html\" title=\"struct melior::dialect::Dialect\">Dialect</a>&lt;'c&gt;","synthetic":true,"types":["melior::dialect::Dialect"]},{"text":"impl Freeze for <a class=\"enum\" href=\"melior/enum.Error.html\" title=\"enum melior::Error\">Error</a>","synthetic":true,"types":["melior::error::Error"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/struct.ExecutionEngine.html\" title=\"struct melior::ExecutionEngine\">ExecutionEngine</a>","synthetic":true,"types":["melior::execution_engine::ExecutionEngine"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.AffineMap.html\" title=\"struct melior::ir::AffineMap\">AffineMap</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::affine_map::AffineMap"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.Attribute.html\" title=\"struct melior::ir::Attribute\">Attribute</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::attribute::Attribute"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/block/struct.Argument.html\" title=\"struct melior::ir::block::Argument\">Argument</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::block::argument::Argument"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/block/struct.Block.html\" title=\"struct melior::ir::block::Block\">Block</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::block::Block"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/block/struct.BlockRef.html\" title=\"struct melior::ir::block::BlockRef\">BlockRef</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::block::BlockRef"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.Identifier.html\" title=\"struct melior::ir::Identifier\">Identifier</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::identifier::Identifier"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.Location.html\" title=\"struct melior::ir::Location\">Location</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::location::Location"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.Module.html\" title=\"struct melior::ir::Module\">Module</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::module::Module"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/operation/struct.Builder.html\" title=\"struct melior::ir::operation::Builder\">Builder</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::operation::builder::Builder"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/operation/struct.ResultValue.html\" title=\"struct melior::ir::operation::ResultValue\">ResultValue</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::operation::result::ResultValue"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/operation/struct.Operation.html\" title=\"struct melior::ir::operation::Operation\">Operation</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::operation::Operation"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/operation/struct.OperationRef.html\" title=\"struct melior::ir::operation::OperationRef\">OperationRef</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::operation::OperationRef"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/ir/struct.Region.html\" title=\"struct melior::ir::Region\">Region</a>","synthetic":true,"types":["melior::ir::region::Region"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.RegionRef.html\" title=\"struct melior::ir::RegionRef\">RegionRef</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::region::RegionRef"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/type/struct.Function.html\" title=\"struct melior::ir::type::Function\">Function</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::type::function::Function"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/ir/type/id/struct.Allocator.html\" title=\"struct melior::ir::type::id::Allocator\">Allocator</a>","synthetic":true,"types":["melior::ir::type::id::allocator::Allocator"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/ir/type/id/struct.Id.html\" title=\"struct melior::ir::type::id::Id\">Id</a>","synthetic":true,"types":["melior::ir::type::id::Id"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/type/struct.Tuple.html\" title=\"struct melior::ir::type::Tuple\">Tuple</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::type::tuple::Tuple"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/ir/type/struct.Type.html\" title=\"struct melior::ir::type::Type\">Type</a>&lt;'c&gt;","synthetic":true,"types":["melior::ir::type::Type"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/ir/struct.Value.html\" title=\"struct melior::ir::Value\">Value</a>&lt;'a&gt;","synthetic":true,"types":["melior::ir::value::Value"]},{"text":"impl&lt;'c&gt; Freeze for <a class=\"struct\" href=\"melior/pass/struct.Manager.html\" title=\"struct melior::pass::Manager\">Manager</a>&lt;'c&gt;","synthetic":true,"types":["melior::pass::manager::Manager"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/pass/struct.OperationManager.html\" title=\"struct melior::pass::OperationManager\">OperationManager</a>&lt;'a&gt;","synthetic":true,"types":["melior::pass::operation_manager::OperationManager"]},{"text":"impl Freeze for <a class=\"struct\" href=\"melior/pass/struct.Pass.html\" title=\"struct melior::pass::Pass\">Pass</a>","synthetic":true,"types":["melior::pass::Pass"]},{"text":"impl&lt;'a&gt; Freeze for <a class=\"struct\" href=\"melior/struct.StringRef.html\" title=\"struct melior::StringRef\">StringRef</a>&lt;'a&gt;","synthetic":true,"types":["melior::string_ref::StringRef"]}];
implementors["mlir_sys"] = [{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.__fsid_t.html\" title=\"struct mlir_sys::__fsid_t\">__fsid_t</a>","synthetic":true,"types":["mlir_sys::__fsid_t"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.max_align_t.html\" title=\"struct mlir_sys::max_align_t\">max_align_t</a>","synthetic":true,"types":["mlir_sys::max_align_t"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirTypeID.html\" title=\"struct mlir_sys::MlirTypeID\">MlirTypeID</a>","synthetic":true,"types":["mlir_sys::MlirTypeID"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirTypeIDAllocator.html\" title=\"struct mlir_sys::MlirTypeIDAllocator\">MlirTypeIDAllocator</a>","synthetic":true,"types":["mlir_sys::MlirTypeIDAllocator"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirStringRef.html\" title=\"struct mlir_sys::MlirStringRef\">MlirStringRef</a>","synthetic":true,"types":["mlir_sys::MlirStringRef"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirLogicalResult.html\" title=\"struct mlir_sys::MlirLogicalResult\">MlirLogicalResult</a>","synthetic":true,"types":["mlir_sys::MlirLogicalResult"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirContext.html\" title=\"struct mlir_sys::MlirContext\">MlirContext</a>","synthetic":true,"types":["mlir_sys::MlirContext"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirDialect.html\" title=\"struct mlir_sys::MlirDialect\">MlirDialect</a>","synthetic":true,"types":["mlir_sys::MlirDialect"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirDialectRegistry.html\" title=\"struct mlir_sys::MlirDialectRegistry\">MlirDialectRegistry</a>","synthetic":true,"types":["mlir_sys::MlirDialectRegistry"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirOperation.html\" title=\"struct mlir_sys::MlirOperation\">MlirOperation</a>","synthetic":true,"types":["mlir_sys::MlirOperation"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirOpPrintingFlags.html\" title=\"struct mlir_sys::MlirOpPrintingFlags\">MlirOpPrintingFlags</a>","synthetic":true,"types":["mlir_sys::MlirOpPrintingFlags"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirBlock.html\" title=\"struct mlir_sys::MlirBlock\">MlirBlock</a>","synthetic":true,"types":["mlir_sys::MlirBlock"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirRegion.html\" title=\"struct mlir_sys::MlirRegion\">MlirRegion</a>","synthetic":true,"types":["mlir_sys::MlirRegion"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirSymbolTable.html\" title=\"struct mlir_sys::MlirSymbolTable\">MlirSymbolTable</a>","synthetic":true,"types":["mlir_sys::MlirSymbolTable"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirAttribute.html\" title=\"struct mlir_sys::MlirAttribute\">MlirAttribute</a>","synthetic":true,"types":["mlir_sys::MlirAttribute"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirIdentifier.html\" title=\"struct mlir_sys::MlirIdentifier\">MlirIdentifier</a>","synthetic":true,"types":["mlir_sys::MlirIdentifier"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirLocation.html\" title=\"struct mlir_sys::MlirLocation\">MlirLocation</a>","synthetic":true,"types":["mlir_sys::MlirLocation"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirModule.html\" title=\"struct mlir_sys::MlirModule\">MlirModule</a>","synthetic":true,"types":["mlir_sys::MlirModule"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirType.html\" title=\"struct mlir_sys::MlirType\">MlirType</a>","synthetic":true,"types":["mlir_sys::MlirType"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirValue.html\" title=\"struct mlir_sys::MlirValue\">MlirValue</a>","synthetic":true,"types":["mlir_sys::MlirValue"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirNamedAttribute.html\" title=\"struct mlir_sys::MlirNamedAttribute\">MlirNamedAttribute</a>","synthetic":true,"types":["mlir_sys::MlirNamedAttribute"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirDialectHandle.html\" title=\"struct mlir_sys::MlirDialectHandle\">MlirDialectHandle</a>","synthetic":true,"types":["mlir_sys::MlirDialectHandle"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirOperationState.html\" title=\"struct mlir_sys::MlirOperationState\">MlirOperationState</a>","synthetic":true,"types":["mlir_sys::MlirOperationState"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirAffineExpr.html\" title=\"struct mlir_sys::MlirAffineExpr\">MlirAffineExpr</a>","synthetic":true,"types":["mlir_sys::MlirAffineExpr"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirAffineMap.html\" title=\"struct mlir_sys::MlirAffineMap\">MlirAffineMap</a>","synthetic":true,"types":["mlir_sys::MlirAffineMap"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirPass.html\" title=\"struct mlir_sys::MlirPass\">MlirPass</a>","synthetic":true,"types":["mlir_sys::MlirPass"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirExternalPass.html\" title=\"struct mlir_sys::MlirExternalPass\">MlirExternalPass</a>","synthetic":true,"types":["mlir_sys::MlirExternalPass"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirPassManager.html\" title=\"struct mlir_sys::MlirPassManager\">MlirPassManager</a>","synthetic":true,"types":["mlir_sys::MlirPassManager"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirOpPassManager.html\" title=\"struct mlir_sys::MlirOpPassManager\">MlirOpPassManager</a>","synthetic":true,"types":["mlir_sys::MlirOpPassManager"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirExternalPassCallbacks.html\" title=\"struct mlir_sys::MlirExternalPassCallbacks\">MlirExternalPassCallbacks</a>","synthetic":true,"types":["mlir_sys::MlirExternalPassCallbacks"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirDiagnostic.html\" title=\"struct mlir_sys::MlirDiagnostic\">MlirDiagnostic</a>","synthetic":true,"types":["mlir_sys::MlirDiagnostic"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirExecutionEngine.html\" title=\"struct mlir_sys::MlirExecutionEngine\">MlirExecutionEngine</a>","synthetic":true,"types":["mlir_sys::MlirExecutionEngine"]},{"text":"impl Freeze for <a class=\"struct\" href=\"mlir_sys/struct.MlirIntegerSet.html\" title=\"struct mlir_sys::MlirIntegerSet\">MlirIntegerSet</a>","synthetic":true,"types":["mlir_sys::MlirIntegerSet"]}];
implementors["once_cell"] = [{"text":"impl&lt;T&gt; !Freeze for <a class=\"struct\" href=\"once_cell/unsync/struct.OnceCell.html\" title=\"struct once_cell::unsync::OnceCell\">OnceCell</a>&lt;T&gt;","synthetic":true,"types":["once_cell::unsync::OnceCell"]},{"text":"impl&lt;T, F&nbsp;=&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.fn.html\">fn</a>() -&gt; T&gt; !Freeze for <a class=\"struct\" href=\"once_cell/unsync/struct.Lazy.html\" title=\"struct once_cell::unsync::Lazy\">Lazy</a>&lt;T, F&gt;","synthetic":true,"types":["once_cell::unsync::Lazy"]},{"text":"impl&lt;T&gt; !Freeze for <a class=\"struct\" href=\"once_cell/sync/struct.OnceCell.html\" title=\"struct once_cell::sync::OnceCell\">OnceCell</a>&lt;T&gt;","synthetic":true,"types":["once_cell::sync::OnceCell"]},{"text":"impl&lt;T, F&nbsp;=&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.fn.html\">fn</a>() -&gt; T&gt; !Freeze for <a class=\"struct\" href=\"once_cell/sync/struct.Lazy.html\" title=\"struct once_cell::sync::Lazy\">Lazy</a>&lt;T, F&gt;","synthetic":true,"types":["once_cell::sync::Lazy"]},{"text":"impl&lt;T&gt; !Freeze for <a class=\"struct\" href=\"once_cell/race/struct.OnceBox.html\" title=\"struct once_cell::race::OnceBox\">OnceBox</a>&lt;T&gt;","synthetic":true,"types":["once_cell::race::once_box::OnceBox"]},{"text":"impl !Freeze for <a class=\"struct\" href=\"once_cell/race/struct.OnceNonZeroUsize.html\" title=\"struct once_cell::race::OnceNonZeroUsize\">OnceNonZeroUsize</a>","synthetic":true,"types":["once_cell::race::OnceNonZeroUsize"]},{"text":"impl !Freeze for <a class=\"struct\" href=\"once_cell/race/struct.OnceBool.html\" title=\"struct once_cell::race::OnceBool\">OnceBool</a>","synthetic":true,"types":["once_cell::race::OnceBool"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()