var sourcesIndex = JSON.parse('{\
"aho_corasick":["",[["nfa",[],["contiguous.rs","mod.rs","noncontiguous.rs"]],["packed",[["teddy",[],["compile.rs","mod.rs","runtime.rs"]]],["api.rs","mod.rs","pattern.rs","rabinkarp.rs","vector.rs"]],["util",[],["alphabet.rs","buffer.rs","byte_frequencies.rs","debug.rs","error.rs","int.rs","mod.rs","prefilter.rs","primitives.rs","remapper.rs","search.rs","special.rs"]]],["ahocorasick.rs","automaton.rs","dfa.rs","lib.rs","macros.rs"]],\
"anes":["",[["sequences",[],["attribute.rs","buffer.rs","color.rs","cursor.rs","terminal.rs"]]],["lib.rs","macros.rs","sequences.rs"]],\
"anstyle":["",[],["color.rs","effect.rs","lib.rs","macros.rs","reset.rs","style.rs"]],\
"bitflags":["",[],["lib.rs"]],\
"cast":["",[],["lib.rs"]],\
"cfg_if":["",[],["lib.rs"]],\
"ciborium":["",[["de",[],["error.rs","mod.rs"]],["ser",[],["error.rs","mod.rs"]],["value",[],["canonical.rs","de.rs","error.rs","integer.rs","mod.rs","ser.rs"]]],["lib.rs","tag.rs"]],\
"ciborium_io":["",[],["lib.rs"]],\
"ciborium_ll":["",[],["dec.rs","enc.rs","hdr.rs","lib.rs","seg.rs"]],\
"clap":["",[],["lib.rs"]],\
"clap_builder":["",[["builder",[],["action.rs","app_settings.rs","arg.rs","arg_group.rs","arg_predicate.rs","arg_settings.rs","command.rs","debug_asserts.rs","ext.rs","mod.rs","os_str.rs","possible_value.rs","range.rs","resettable.rs","str.rs","styled_str.rs","value_hint.rs","value_parser.rs"]],["error",[],["format.rs","kind.rs","mod.rs"]],["output",[],["fmt.rs","help.rs","mod.rs","usage.rs"]],["parser",[["features",[],["mod.rs","suggestions.rs"]],["matches",[],["arg_matches.rs","matched_arg.rs","mod.rs","value_source.rs"]]],["arg_matcher.rs","error.rs","mod.rs","parser.rs","validator.rs"]],["util",[],["any_value.rs","color.rs","flat_map.rs","flat_set.rs","graph.rs","id.rs","mod.rs","str_to_bool.rs"]]],["derive.rs","lib.rs","macros.rs","mkeymap.rs"]],\
"clap_lex":["",[],["ext.rs","lib.rs"]],\
"convert_case":["",[],["case.rs","converter.rs","lib.rs","pattern.rs","segmentation.rs"]],\
"criterion":["",[["analysis",[],["compare.rs","mod.rs"]],["html",[],["mod.rs"]],["plot",[["gnuplot_backend",[],["distributions.rs","iteration_times.rs","mod.rs","pdf.rs","regression.rs","summary.rs","t_test.rs"]],["plotters_backend",[],["distributions.rs","iteration_times.rs","mod.rs","pdf.rs","regression.rs","summary.rs","t_test.rs"]]],["mod.rs"]],["stats",[["bivariate",[],["bootstrap.rs","mod.rs","regression.rs","resamples.rs"]],["univariate",[["kde",[],["kernel.rs","mod.rs"]],["outliers",[],["mod.rs","tukey.rs"]]],["bootstrap.rs","mixed.rs","mod.rs","percentiles.rs","resamples.rs","sample.rs"]]],["float.rs","mod.rs","rand_util.rs","tuple.rs"]]],["async_executor.rs","bencher.rs","benchmark.rs","benchmark_group.rs","connection.rs","error.rs","estimate.rs","format.rs","fs.rs","kde.rs","lib.rs","macros.rs","macros_private.rs","measurement.rs","profiler.rs","report.rs","routine.rs"]],\
"criterion_plot":["",[],["axis.rs","candlestick.rs","curve.rs","data.rs","display.rs","errorbar.rs","filledcurve.rs","grid.rs","key.rs","lib.rs","map.rs","prelude.rs","proxy.rs","traits.rs"]],\
"crossbeam_channel":["",[["flavors",[],["array.rs","at.rs","list.rs","mod.rs","never.rs","tick.rs","zero.rs"]]],["channel.rs","context.rs","counter.rs","err.rs","lib.rs","select.rs","select_macro.rs","utils.rs","waker.rs"]],\
"crossbeam_deque":["",[],["deque.rs","lib.rs"]],\
"crossbeam_epoch":["",[["sync",[],["list.rs","mod.rs","once_lock.rs","queue.rs"]]],["atomic.rs","collector.rs","default.rs","deferred.rs","epoch.rs","guard.rs","internal.rs","lib.rs"]],\
"crossbeam_utils":["",[["atomic",[],["atomic_cell.rs","consume.rs","mod.rs","seq_lock.rs"]],["sync",[],["mod.rs","once_lock.rs","parker.rs","sharded_lock.rs","wait_group.rs"]]],["backoff.rs","cache_padded.rs","lib.rs","thread.rs"]],\
"dashmap":["",[["mapref",[],["entry.rs","mod.rs","multiple.rs","one.rs"]],["setref",[],["mod.rs","multiple.rs","one.rs"]]],["iter.rs","iter_set.rs","lib.rs","lock.rs","read_only.rs","set.rs","t.rs","try_result.rs","util.rs"]],\
"either":["",[],["lib.rs"]],\
"half":["",[["bfloat",[],["convert.rs"]],["binary16",[],["convert.rs"]]],["bfloat.rs","binary16.rs","lib.rs","slice.rs"]],\
"hashbrown":["",[["external_trait_impls",[],["mod.rs"]],["raw",[],["alloc.rs","bitmask.rs","mod.rs","sse2.rs"]]],["lib.rs","macros.rs","map.rs","scopeguard.rs","set.rs"]],\
"io_lifetimes":["",[],["example_ffi.rs","lib.rs","portability.rs","raw.rs","traits.rs","views.rs"]],\
"is_terminal":["",[],["lib.rs"]],\
"itertools":["",[["adaptors",[],["coalesce.rs","map.rs","mod.rs","multi_product.rs"]]],["combinations.rs","combinations_with_replacement.rs","concat_impl.rs","cons_tuples_impl.rs","diff.rs","duplicates_impl.rs","either_or_both.rs","exactly_one_err.rs","extrema_set.rs","flatten_ok.rs","format.rs","free.rs","group_map.rs","groupbylazy.rs","grouping_map.rs","impl_macros.rs","intersperse.rs","k_smallest.rs","kmerge_impl.rs","lazy_buffer.rs","lib.rs","merge_join.rs","minmax.rs","multipeek_impl.rs","pad_tail.rs","peek_nth.rs","peeking_take_while.rs","permutations.rs","powerset.rs","process_results_impl.rs","put_back_n_impl.rs","rciter_impl.rs","repeatn.rs","size_hint.rs","sources.rs","tee.rs","tuple_impl.rs","unique_impl.rs","unziptuple.rs","with_position.rs","zip_eq_impl.rs","zip_longest.rs","ziptuple.rs"]],\
"itoa":["",[],["lib.rs","udiv128.rs"]],\
"libc":["",[["unix",[["linux_like",[["linux",[["arch",[["generic",[],["mod.rs"]]],["mod.rs"]],["gnu",[["b64",[["x86_64",[],["align.rs","mod.rs","not_x32.rs"]]],["mod.rs"]]],["align.rs","mod.rs"]]],["align.rs","mod.rs","non_exhaustive.rs"]]],["mod.rs"]]],["align.rs","mod.rs"]]],["fixed_width_ints.rs","lib.rs","macros.rs"]],\
"linux_raw_sys":["",[["x86_64",[],["errno.rs","general.rs","ioctl.rs"]]],["lib.rs"]],\
"lock_api":["",[],["lib.rs","mutex.rs","remutex.rs","rwlock.rs"]],\
"melior":["",[["diagnostic",[],["handler_id.rs","severity.rs"]],["dialect",[["llvm",[],["attributes.rs","type.rs"]]],["arith.rs","cf.rs","func.rs","handle.rs","index.rs","llvm.rs","memref.rs","registry.rs","scf.rs"]],["ir",[["attribute",[],["array.rs","attribute_like.rs","dense_elements.rs","dense_i32_array.rs","dense_i64_array.rs","flat_symbol_ref.rs","float.rs","integer.rs","macro.rs","string.rs","type.rs"]],["block",[],["argument.rs"]],["operation",[],["builder.rs","printing_flags.rs","result.rs"]],["type",[["id",[],["allocator.rs"]]],["function.rs","id.rs","integer.rs","macro.rs","mem_ref.rs","ranked_tensor.rs","tuple.rs","type_like.rs"]],["value",[],["value_like.rs"]]],["affine_map.rs","attribute.rs","block.rs","identifier.rs","location.rs","module.rs","operation.rs","region.rs","type.rs","value.rs"]],["pass",[],["async.rs","conversion.rs","gpu.rs","linalg.rs","manager.rs","operation_manager.rs","sparse_tensor.rs","transform.rs"]]],["context.rs","diagnostic.rs","dialect.rs","error.rs","execution_engine.rs","ir.rs","lib.rs","logical_result.rs","macro.rs","pass.rs","string_ref.rs","utility.rs"]],\
"melior_macro":["",[],["attribute.rs","lib.rs","operation.rs","parse.rs","pass.rs","type.rs","utility.rs"]],\
"memchr":["",[["memchr",[["x86",[],["avx.rs","mod.rs","sse2.rs"]]],["fallback.rs","iter.rs","mod.rs","naive.rs"]],["memmem",[["prefilter",[["x86",[],["avx.rs","mod.rs","sse.rs"]]],["fallback.rs","genericsimd.rs","mod.rs"]],["x86",[],["avx.rs","mod.rs","sse.rs"]]],["byte_frequencies.rs","genericsimd.rs","mod.rs","rabinkarp.rs","rarebytes.rs","twoway.rs","util.rs","vector.rs"]]],["cow.rs","lib.rs"]],\
"memoffset":["",[],["lib.rs","offset_of.rs","raw_field.rs","span_of.rs"]],\
"mlir_sys":["",[],["lib.rs"]],\
"num_cpus":["",[],["lib.rs","linux.rs"]],\
"num_traits":["",[["ops",[],["checked.rs","euclid.rs","inv.rs","mod.rs","mul_add.rs","overflowing.rs","saturating.rs","wrapping.rs"]]],["bounds.rs","cast.rs","float.rs","identities.rs","int.rs","lib.rs","macros.rs","pow.rs","real.rs","sign.rs"]],\
"once_cell":["",[],["imp_std.rs","lib.rs","race.rs"]],\
"oorandom":["",[],["lib.rs"]],\
"parking_lot_core":["",[["thread_parker",[],["linux.rs","mod.rs"]]],["lib.rs","parking_lot.rs","spinwait.rs","util.rs","word_lock.rs"]],\
"plotters":["",[["chart",[["context",[["cartesian2d",[],["draw_impl.rs","mod.rs"]],["cartesian3d",[],["draw_impl.rs","mod.rs"]]]]],["axes3d.rs","builder.rs","context.rs","dual_coord.rs","mesh.rs","mod.rs","series.rs","state.rs"]],["coord",[["ranged1d",[["combinators",[],["ckps.rs","group_by.rs","linspace.rs","logarithmic.rs","mod.rs","nested.rs","partial_axis.rs"]],["types",[],["mod.rs","numeric.rs","slice.rs"]]],["discrete.rs","mod.rs"]],["ranged2d",[],["cartesian.rs","mod.rs"]],["ranged3d",[],["cartesian3d.rs","mod.rs","projection.rs"]]],["mod.rs","translate.rs"]],["data",[],["data_range.rs","float.rs","mod.rs","quartiles.rs"]],["drawing",[["backend_impl",[],["mod.rs"]]],["area.rs","mod.rs"]],["element",[],["basic_shapes.rs","basic_shapes_3d.rs","composable.rs","dynelem.rs","mod.rs","pie.rs","points.rs","text.rs"]],["series",[],["area_series.rs","line_series.rs","mod.rs"]],["style",[["colors",[],["mod.rs"]],["font",[],["font_desc.rs","mod.rs","naive.rs"]]],["color.rs","mod.rs","palette.rs","shape.rs","size.rs","text.rs"]]],["lib.rs"]],\
"plotters_backend":["",[["rasterizer",[],["circle.rs","line.rs","mod.rs","path.rs","polygon.rs","rect.rs"]]],["lib.rs","style.rs","text.rs"]],\
"plotters_svg":["",[],["lib.rs","svg.rs"]],\
"proc_macro2":["",[],["detection.rs","extra.rs","fallback.rs","lib.rs","marker.rs","parse.rs","rcvec.rs","wrapper.rs"]],\
"quote":["",[],["ext.rs","format.rs","ident_fragment.rs","lib.rs","runtime.rs","spanned.rs","to_tokens.rs"]],\
"rayon":["",[["collections",[],["binary_heap.rs","btree_map.rs","btree_set.rs","hash_map.rs","hash_set.rs","linked_list.rs","mod.rs","vec_deque.rs"]],["compile_fail",[],["cannot_collect_filtermap_data.rs","cannot_zip_filtered_data.rs","cell_par_iter.rs","mod.rs","must_use.rs","no_send_par_iter.rs","rc_par_iter.rs"]],["iter",[["collect",[],["consumer.rs","mod.rs"]],["find_first_last",[],["mod.rs"]],["plumbing",[],["mod.rs"]]],["chain.rs","chunks.rs","cloned.rs","copied.rs","empty.rs","enumerate.rs","extend.rs","filter.rs","filter_map.rs","find.rs","flat_map.rs","flat_map_iter.rs","flatten.rs","flatten_iter.rs","fold.rs","fold_chunks.rs","fold_chunks_with.rs","for_each.rs","from_par_iter.rs","inspect.rs","interleave.rs","interleave_shortest.rs","intersperse.rs","len.rs","map.rs","map_with.rs","mod.rs","multizip.rs","noop.rs","once.rs","panic_fuse.rs","par_bridge.rs","positions.rs","product.rs","reduce.rs","repeat.rs","rev.rs","skip.rs","skip_any.rs","skip_any_while.rs","splitter.rs","step_by.rs","sum.rs","take.rs","take_any.rs","take_any_while.rs","try_fold.rs","try_reduce.rs","try_reduce_with.rs","unzip.rs","update.rs","while_some.rs","zip.rs","zip_eq.rs"]],["slice",[],["chunks.rs","mergesort.rs","mod.rs","quicksort.rs","rchunks.rs"]]],["array.rs","delegate.rs","lib.rs","math.rs","option.rs","par_either.rs","prelude.rs","private.rs","range.rs","range_inclusive.rs","result.rs","split_producer.rs","str.rs","string.rs","vec.rs"]],\
"rayon_core":["",[["broadcast",[],["mod.rs"]],["compile_fail",[],["mod.rs","quicksort_race1.rs","quicksort_race2.rs","quicksort_race3.rs","rc_return.rs","rc_upvar.rs","scope_join_bad.rs"]],["join",[],["mod.rs"]],["scope",[],["mod.rs"]],["sleep",[],["counters.rs","mod.rs"]],["spawn",[],["mod.rs"]],["thread_pool",[],["mod.rs"]]],["job.rs","latch.rs","lib.rs","log.rs","private.rs","registry.rs","unwind.rs"]],\
"regex":["",[["literal",[],["imp.rs","mod.rs"]]],["backtrack.rs","compile.rs","dfa.rs","error.rs","exec.rs","expand.rs","find_byte.rs","input.rs","lib.rs","pikevm.rs","pool.rs","prog.rs","re_builder.rs","re_bytes.rs","re_set.rs","re_trait.rs","re_unicode.rs","sparse.rs","utf8.rs"]],\
"regex_syntax":["",[["ast",[],["mod.rs","parse.rs","print.rs","visitor.rs"]],["hir",[],["interval.rs","literal.rs","mod.rs","print.rs","translate.rs","visitor.rs"]],["unicode_tables",[],["age.rs","case_folding_simple.rs","general_category.rs","grapheme_cluster_break.rs","mod.rs","perl_word.rs","property_bool.rs","property_names.rs","property_values.rs","script.rs","script_extension.rs","sentence_break.rs","word_break.rs"]]],["debug.rs","either.rs","error.rs","lib.rs","parser.rs","rank.rs","unicode.rs","utf8.rs"]],\
"rustix":["",[["backend",[["linux_raw",[["arch",[["inline",[],["mod.rs","x86_64.rs"]]],["mod.rs"]],["io",[],["epoll.rs","errno.rs","mod.rs","poll_fd.rs","syscalls.rs","types.rs"]],["process",[],["cpu_set.rs","mod.rs","syscalls.rs","types.rs","wait.rs"]],["termios",[],["mod.rs","syscalls.rs","types.rs"]],["time",[],["mod.rs","types.rs"]]],["c.rs","conv.rs","elf.rs","mod.rs","reg.rs"]]]],["ffi",[],["mod.rs"]],["io",[],["close.rs","dup.rs","errno.rs","eventfd.rs","fcntl.rs","ioctl.rs","is_read_write.rs","mod.rs","pipe.rs","poll.rs","read_write.rs","seek_from.rs","stdio.rs"]],["path",[],["arg.rs","mod.rs"]],["process",[],["chdir.rs","chroot.rs","exit.rs","id.rs","kill.rs","membarrier.rs","mod.rs","pidfd.rs","prctl.rs","priority.rs","rlimit.rs","sched.rs","sched_yield.rs","system.rs","umask.rs","wait.rs"]],["termios",[],["cf.rs","constants.rs","mod.rs","tc.rs","tty.rs"]]],["const_assert.rs","cstr.rs","lib.rs","utils.rs","weak.rs"]],\
"ryu":["",[["buffer",[],["mod.rs"]],["pretty",[],["exponent.rs","mantissa.rs","mod.rs"]]],["common.rs","d2s.rs","d2s_full_table.rs","d2s_intrinsics.rs","digit_table.rs","f2s.rs","f2s_intrinsics.rs","lib.rs"]],\
"same_file":["",[],["lib.rs","unix.rs"]],\
"scopeguard":["",[],["lib.rs"]],\
"serde":["",[["de",[],["format.rs","ignored_any.rs","impls.rs","mod.rs","seed.rs","utf8.rs","value.rs"]],["private",[],["de.rs","doc.rs","mod.rs","ser.rs","size_hint.rs"]],["ser",[],["fmt.rs","impls.rs","impossible.rs","mod.rs"]]],["integer128.rs","lib.rs","macros.rs"]],\
"serde_derive":["",[["internals",[],["ast.rs","attr.rs","case.rs","check.rs","ctxt.rs","mod.rs","receiver.rs","respan.rs","symbol.rs"]]],["bound.rs","de.rs","dummy.rs","fragment.rs","lib.rs","pretend.rs","ser.rs","this.rs","try.rs"]],\
"serde_json":["",[["features_check",[],["mod.rs"]],["io",[],["mod.rs"]],["value",[],["de.rs","from.rs","index.rs","mod.rs","partial_eq.rs","ser.rs"]]],["de.rs","error.rs","iter.rs","lib.rs","macros.rs","map.rs","number.rs","read.rs","ser.rs"]],\
"smallvec":["",[],["lib.rs"]],\
"syn":["",[["gen",[],["clone.rs","debug.rs","eq.rs","hash.rs","visit.rs","visit_mut.rs"]]],["attr.rs","bigint.rs","buffer.rs","custom_keyword.rs","custom_punctuation.rs","data.rs","derive.rs","discouraged.rs","drops.rs","error.rs","export.rs","expr.rs","ext.rs","file.rs","gen_helper.rs","generics.rs","group.rs","ident.rs","item.rs","lib.rs","lifetime.rs","lit.rs","lookahead.rs","mac.rs","macros.rs","meta.rs","op.rs","parse.rs","parse_macro_input.rs","parse_quote.rs","pat.rs","path.rs","print.rs","punctuated.rs","restriction.rs","sealed.rs","span.rs","spanned.rs","stmt.rs","thread.rs","token.rs","tt.rs","ty.rs","verbatim.rs","whitespace.rs"]],\
"tinytemplate":["",[],["compiler.rs","error.rs","instruction.rs","lib.rs","syntax.rs","template.rs"]],\
"unicode_ident":["",[],["lib.rs","tables.rs"]],\
"unicode_segmentation":["",[],["grapheme.rs","lib.rs","sentence.rs","tables.rs","word.rs"]],\
"walkdir":["",[],["dent.rs","error.rs","lib.rs","util.rs"]]\
}');
createSourceSidebar();
