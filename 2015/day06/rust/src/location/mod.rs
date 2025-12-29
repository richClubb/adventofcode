
#[derive(Debug, PartialEq)]
pub struct Location {
    pub x_pos: usize,
    pub y_pos: usize
}

impl Location {
    pub fn new(new_str: String) -> Self {
        let results: Vec<usize> = new_str.split(",").into_iter().map(|val| val.parse::<usize>().unwrap()).collect();
        Self {x_pos: results[0], y_pos: results[1]}
    }
}