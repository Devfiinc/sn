use ndarray::*;

pub struct LR {
    _id : String
}

impl LR {
    pub fn new(id: &str) -> LR{
        LR {
            _id : id.to_string()
        }
    }

    pub fn get_id(&self) -> String {
        format!("{}", self._id)
    }
}