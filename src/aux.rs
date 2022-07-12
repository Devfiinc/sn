trait TupleAppend<T> {
    type ResultType;

    fn append(self, t: T) -> Self::ResultType;
}

impl<T> TupleAppend<T> for () {
    type ResultType = (T,);

    fn append(self, t: T) -> Self::ResultType {
        (t,)
    }
}

macro_rules! impl_tuple_append {
    ( () ) => {};
    ( ( $t0:ident $(, $types:ident)* ) ) => {
        impl<$t0, $($types,)* T> TupleAppend<T> for ($t0, $($types,)*) {
            // Trailing comma, just to be extra sure we are dealing
            // with a tuple and not a parenthesized type/expr.
            type ResultType = ($t0, $($types,)* T,);

            fn append(self, t: T) -> Self::ResultType {
                // Reuse the type identifiers to destructure ourselves:
                let ($t0, $($types,)*) = self;
                // Create a new tuple with the original elements, plus the new one:
                ($t0, $($types,)* t,)
            }
        }

        // Recurse for one smaller size:
        impl_tuple_append! { ($($types),*) }
    };
}

impl_tuple_append! {
    // Supports tuples up to size 99:
    (      _1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,
     _10, _11, _12, _13, _14, _15, _16, _17, _18, _19,
     _20, _21, _22, _23, _24, _25, _26, _27, _28, _29,
     _30, _31, _32, _33, _34, _35, _36, _37, _38, _39,
     _40, _41, _42, _43, _44, _45, _46, _47, _48, _49,
     _50, _51, _52, _53, _54, _55, _56, _57, _58, _59,
     _60, _61, _62, _63, _64, _65, _66, _67, _68, _69,
     _70, _71, _72, _73, _74, _75, _76, _77, _78, _79,
     _80, _81, _82, _83, _84, _85, _86, _87, _88, _89,
     _90, _91, _92, _93, _94, _95, _96, _97, _98, _99)
}