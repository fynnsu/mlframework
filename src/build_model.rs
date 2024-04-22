#[macro_export]
macro_rules! build_mod {
    ($mod_name:ident inputs=[$($in_name:ident : $in_type:ty),+ ], outputs=[$($out_name:ident : $out_type:ty),+ ]) => {

        pub struct $mod_name {
        $(pub $in_name : $in_type), +,
        $(pub $out_name: $out_type),+
        }

        impl $mod_name {
            pub fn new($($in_name : $in_type),+, $($out_name : $out_type),+) -> Self {
                Self {
                    $($in_name),+,
                $($out_name),+
                }
            }

            pub fn recompute(&self, $($in_name: Vec<<$in_type as mlframework::tensor::HasDtype>::Dtype>),+) {
                $(
                    self.$in_name.replace_data_with($in_name)
                );+;
                $(
                    self.$out_name.recompute()
                );+;
            }
        }
    };
}

// todo: Might replace above with something like make_mod_placeholders and only store the inputs.
// Since the main usage for this seems to be a simpler interface for replacing the input values.
