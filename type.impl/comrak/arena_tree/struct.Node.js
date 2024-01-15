(function() {var type_impls = {
"comrak":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Node%3C'a,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#62-249\">source</a><a href=\"#impl-Node%3C'a,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, T&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.new\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#67-76\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.new\" class=\"fn\">new</a>(data: T) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;</h4></section></summary><div class=\"docblock\"><p>Create a new node from its associated data.</p>\n<p>Typically, this node needs to be moved into an arena allocator\nbefore it can be used in a tree.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.parent\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#79-81\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.parent\" class=\"fn\">parent</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Return a reference to the parent node, unless this node is the root of the tree.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.first_child\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#84-86\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.first_child\" class=\"fn\">first_child</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Return a reference to the first child of this node, unless it has no child.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.last_child\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#89-91\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.last_child\" class=\"fn\">last_child</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Return a reference to the last child of this node, unless it has no child.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.previous_sibling\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#94-96\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.previous_sibling\" class=\"fn\">previous_sibling</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Return a reference to the previous sibling of this node, unless it is a first child.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.next_sibling\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#99-101\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.next_sibling\" class=\"fn\">next_sibling</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Return a reference to the next sibling of this node, unless it is a last child.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.same_node\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#104-106\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.same_node\" class=\"fn\">same_node</a>(&amp;self, other: &amp;<a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class=\"docblock\"><p>Returns whether two references point to the same node.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ancestors\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#111-113\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.ancestors\" class=\"fn\">ancestors</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Ancestors.html\" title=\"struct comrak::arena_tree::Ancestors\">Ancestors</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"Ancestors&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and its ancestors.</p>\n<p>Call <code>.next().unwrap()</code> once on the iterator to skip the node itself.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.preceding_siblings\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#118-120\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.preceding_siblings\" class=\"fn\">preceding_siblings</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.PrecedingSiblings.html\" title=\"struct comrak::arena_tree::PrecedingSiblings\">PrecedingSiblings</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"PrecedingSiblings&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and the siblings before it.</p>\n<p>Call <code>.next().unwrap()</code> once on the iterator to skip the node itself.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.following_siblings\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#125-127\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.following_siblings\" class=\"fn\">following_siblings</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.FollowingSiblings.html\" title=\"struct comrak::arena_tree::FollowingSiblings\">FollowingSiblings</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"FollowingSiblings&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and the siblings after it.</p>\n<p>Call <code>.next().unwrap()</code> once on the iterator to skip the node itself.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.children\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#130-132\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.children\" class=\"fn\">children</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Children.html\" title=\"struct comrak::arena_tree::Children\">Children</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"Children&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node’s children.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.reverse_children\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#135-137\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.reverse_children\" class=\"fn\">reverse_children</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.ReverseChildren.html\" title=\"struct comrak::arena_tree::ReverseChildren\">ReverseChildren</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"ReverseChildren&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node’s children, in reverse order.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.descendants\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#143-145\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.descendants\" class=\"fn\">descendants</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Descendants.html\" title=\"struct comrak::arena_tree::Descendants\">Descendants</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"Descendants&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and its descendants, in tree order.</p>\n<p>Parent nodes appear before the descendants.\nCall <code>.next().unwrap()</code> once on the iterator to skip the node itself.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.traverse\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#148-153\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.traverse\" class=\"fn\">traverse</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.Traverse.html\" title=\"struct comrak::arena_tree::Traverse\">Traverse</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"Traverse&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and its descendants, in tree order.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.reverse_traverse\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#156-161\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.reverse_traverse\" class=\"fn\">reverse_traverse</a>(&amp;'a self) -&gt; <a class=\"struct\" href=\"comrak/arena_tree/struct.ReverseTraverse.html\" title=\"struct comrak::arena_tree::ReverseTraverse\">ReverseTraverse</a>&lt;'a, T&gt; <a href=\"#\" class=\"tooltip\" data-notable-ty=\"ReverseTraverse&lt;&#39;a, T&gt;\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Return an iterator of references to this node and its descendants, in tree order.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.detach\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#164-180\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.detach\" class=\"fn\">detach</a>(&amp;self)</h4></section></summary><div class=\"docblock\"><p>Detach a node from its parent and siblings. Children are not affected.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.append\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#183-195\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.append\" class=\"fn\">append</a>(&amp;'a self, new_child: &amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;)</h4></section></summary><div class=\"docblock\"><p>Append a new child to this node, after existing children.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.prepend\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#198-210\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.prepend\" class=\"fn\">prepend</a>(&amp;'a self, new_child: &amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;)</h4></section></summary><div class=\"docblock\"><p>Prepend a new child to this node, before existing children.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.insert_after\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#213-229\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.insert_after\" class=\"fn\">insert_after</a>(&amp;'a self, new_sibling: &amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;)</h4></section></summary><div class=\"docblock\"><p>Insert a new sibling after this node.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.insert_before\" class=\"method\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#232-248\">source</a><h4 class=\"code-header\">pub fn <a href=\"comrak/arena_tree/struct.Node.html#tymethod.insert_before\" class=\"fn\">insert_before</a>(&amp;'a self, new_sibling: &amp;'a <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;)</h4></section></summary><div class=\"docblock\"><p>Insert a new sibling before this node.</p>\n</div></details></div></details>",0,"comrak::nodes::AstNode"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-Node%3C'a,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#37-60\">source</a><a href=\"#impl-Debug-for-Node%3C'a,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"comrak/arena_tree/struct.Node.html\" title=\"struct comrak::arena_tree::Node\">Node</a>&lt;'a, T&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> + 'a,</span></h3></section></summary><div class=\"docblock\"><p>A simple Debug implementation that prints the children as a tree, without\nlooping through the various interior pointer cycles.</p>\n</div><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/comrak/arena_tree.rs.html#41-59\">source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/struct.Error.html\" title=\"struct core::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","comrak::nodes::AstNode"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()